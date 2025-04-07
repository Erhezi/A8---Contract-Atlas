import pip_system_certs.wrapt_requests
import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
from flask import current_app
from werkzeug.utils import secure_filename
from contextlib import contextmanager
import pip_system_certs
from scipy.spatial.distance import cosine

# model cache for sentence transformers
# This is a simple cache to avoid reloading the model multiple times
_MODEL_CACHE = {}

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'csv'}

def read_file(file_path):
    """Read the uploaded file into a pandas DataFrame"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, dtype=str)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path, dtype=str)
    return None

def get_file_headers(file_path):
    """Get the headers from the uploaded file"""
    df = read_file(file_path)
    if df is not None:
        return df.columns.tolist()
    return []

def reduce_mfg_part_num(mfg_part_num):
    """Simplify manufacturer part number by removing dashes and leading zeros"""
    if pd.isnull(mfg_part_num) or mfg_part_num.strip() == '':
        return np.nan
    mfg_part_num = mfg_part_num.strip().replace('-', '').strip()
    if mfg_part_num.isdigit():
        return str(int(mfg_part_num))
    return mfg_part_num

def strip_columns(df):
    """Strip whitespace from all string columns in the DataFrame"""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    return df

def prepare_dataframe(df, column_mapping):
    """Map columns and prepare dataframe for validation"""
    # Required fields (all required except 'Buyer Part Num')
    required_fields = [
        'Mfg Part Num', 'Vendor Part Num', 'Description', 
        'Contract Price', 'UOM', 'QOE', 'Effective Date', 'Expiration Date', 
        'Contract Number', 'ERP Vendor ID'
    ]
    
    # Create a copy of the dataframe with standard field names
    mapped_df = df.copy()
    mapped_df = strip_columns(mapped_df)
    
    # Rename columns according to the mapping
    for std_field, user_field in column_mapping.items():
        if user_field in mapped_df.columns:
            mapped_df.rename(columns={user_field: std_field}, inplace=True)
        if 'Buyer Part Num' not in mapped_df.columns:
            mapped_df.loc[:, 'Buyer Part Num'] = ''

    # Add derived columns
    mapped_df['Original UOM'] = mapped_df['UOM'].apply(lambda x: np.nan if pd.isnull(x) else x.strip().upper())
    mapped_df['Reduced Mfg Part Num'] = mapped_df['Mfg Part Num'].apply(reduce_mfg_part_num)
    mapped_df['File Row'] = [i for i in range(1, len(mapped_df) + 1)]
    
    # Check for missing required fields
    missing_fields = [field for field in required_fields if field not in mapped_df.columns]
    if missing_fields:
        return None, None, missing_fields, required_fields
    
    # Create copies for validation and results
    result_df = mapped_df[required_fields[:2] + ['Buyer Part Num'] + required_fields[2:] + ['Reduced Mfg Part Num', 'Original UOM', 'File Row']].copy()
    error_df = result_df.copy()
    
    # Add error columns
    error_df['Error-Missing Field'] = ''
    error_df['Error-Invalid Date'] = ''
    error_df['Error-Invalid Price'] = ''
    error_df['Error-Invalid QOE'] = ''
    error_df['Error-Invalid UOM'] = ''
    error_df['Error-EA QOE NOT 1'] = ''
    error_df['Error-Multiple Vendors'] = '' 
    error_df['Warning-Potential Duplicates'] = ''
    error_df['Has Error'] = False
    
    return error_df, result_df, missing_fields, required_fields

def validate_required_fields(error_df, required_fields):
    """Check that all required fields have values"""
    for field in required_fields:
        missing_mask = error_df[field].isna() | (error_df[field].str.strip() == '')
        error_df.loc[missing_mask, 'Error-Missing Field'] = 'Missing Data'
        error_df.loc[missing_mask, 'Has Error'] = True
    return error_df

def validate_dates(error_df):
    """Validate date fields format and logic"""
    today_dt = datetime.now().date()

    # Convert date strings to datetime objects
    error_df['Effective_Date_Dt'] = pd.to_datetime(error_df['Effective Date'], errors='coerce')
    error_df['Expiration_Date_Dt'] = pd.to_datetime(error_df['Expiration Date'], errors='coerce')

    # Create YYYY-MM-DD string versions for the output
    error_df['Effective Date'] = error_df['Effective_Date_Dt'].dt.strftime('%Y-%m-%d')
    error_df['Expiration Date'] = error_df['Expiration_Date_Dt'].dt.strftime('%Y-%m-%d')

    # Mark rows with unparseable dates
    eff_date_mask = error_df['Effective_Date_Dt'].isna() & ~(error_df['Effective Date'].isna() | (error_df['Effective Date'].str.strip() == ''))
    error_df.loc[eff_date_mask, 'Error-Invalid Date'] = 'Invalid Effective Date format'
    error_df.loc[eff_date_mask, 'Has Error'] = True

    exp_date_mask = error_df['Expiration_Date_Dt'].isna() & ~(error_df['Expiration Date'].isna() | (error_df['Expiration Date'].str.strip() == ''))
    error_df.loc[exp_date_mask, 'Error-Invalid Date'] = 'Invalid Expiration Date format'
    error_df.loc[exp_date_mask, 'Has Error'] = True

    # Skip validation for already marked error rows
    valid_dates_mask = (~error_df['Effective_Date_Dt'].isna()) & (~error_df['Expiration_Date_Dt'].isna())

    # Validate expiration date >= today and >= effective date
    invalid_exp_mask = valid_dates_mask & (
        (error_df['Expiration_Date_Dt'].dt.date < today_dt) | 
        (error_df['Expiration_Date_Dt'] < error_df['Effective_Date_Dt'])
    )
    error_df.loc[invalid_exp_mask, 'Error-Invalid Date'] = 'Expiration Date must be >= Effective Date and today'
    error_df.loc[invalid_exp_mask, 'Has Error'] = True
    
    return error_df

def convert_price(price_str):
    """Convert price string to float"""
    if pd.isna(price_str) or price_str.strip() == '':
        return None
    try:
        # Remove $ and commas
        cleaned = re.sub(r'[,$]', '', price_str.strip())
        return float(cleaned)
    except:
        return None

def validate_prices(error_df):
    """Validate contract prices"""
    error_df['Price_Parsed'] = error_df['Contract Price'].apply(convert_price)
    error_df.loc[error_df['Price_Parsed'].isna(), 'Error-Invalid Price'] = 'Price not Recognized'
    error_df.loc[error_df['Price_Parsed'].isna(), 'Has Error'] = True
    return error_df

def convert_qoe(qoe_str):
    """Convert QOE string to integer"""
    if pd.isna(qoe_str) or qoe_str.strip() == '':
        return None
    try:
        return int(qoe_str.strip())
    except:
        return None

def validate_qoe(error_df):
    """Validate Quantity of Each (QOE)"""
    error_df['QOE_Parsed'] = error_df['QOE'].apply(convert_qoe)
    error_df.loc[error_df['QOE_Parsed'].isna(), 'Error-Invalid QOE'] = 'QOE not Recognized'
    error_df.loc[error_df['QOE_Parsed'].isna(), 'Has Error'] = True
    return error_df

def validate_uom(error_df):
    """Validate Units of Measure (UOM)"""
    try:
        uom_file_path = os.path.join(current_app.root_path, 'data', 'UOM.csv')
        uom_df = pd.read_csv(uom_file_path)
        
        # Create UOM dictionary mapping
        uom_dict = dict(zip(uom_df['see UOM'].str.upper(), uom_df['use UOM']))
        
        # Check if UOM values exist in the dictionary
        invalid_uom_mask = ~error_df['Original UOM'].isin(uom_dict.keys())
        
        # Mark rows with invalid UOM values
        error_df.loc[invalid_uom_mask, 'Error-Invalid UOM'] = 'UOM not recognized'
        error_df.loc[invalid_uom_mask, 'Has Error'] = True
        
        # For valid UOMs, map to the standardized value
        valid_uom_mask = ~invalid_uom_mask
        error_df.loc[valid_uom_mask, 'UOM'] = error_df.loc[valid_uom_mask, 'Original UOM'].map(uom_dict)
        
        # Set invalid UOMs to NaN
        error_df.loc[invalid_uom_mask, 'UOM'] = np.nan
        
    except Exception as e:
        print(f"Error loading or processing UOM file: {str(e)}")
    
    return error_df

def validate_uom_qoe_compatibility(error_df):
    """Validate UOM-QOE compatibility (EA units must have QOE=1)"""
    error_df.loc[(error_df['UOM'] == 'EA') & (error_df['QOE_Parsed'] != 1), 'Error-EA QOE NOT 1'] = 'QOE must be 1 for UOM EA'
    error_df.loc[(error_df['UOM'] == 'EA') & (error_df['QOE_Parsed'] != 1), 'Has Error'] = True
    return error_df


def validate_contract_vendor_relationship(error_df):
    """Validate that each Contract Number has only one ERP Vendor ID"""
    # Group by Contract Number and get unique ERP Vendor IDs for each
    contract_vendors = error_df.groupby('Contract Number')['ERP Vendor ID'].unique()
    
    # Find contracts with multiple vendors
    invalid_contracts = contract_vendors[contract_vendors.apply(len) > 1]
    
    if len(invalid_contracts) > 0:
        # For each invalid contract, mark all its rows
        for contract in invalid_contracts.index:
            # Create mask for all rows with this contract
            contract_mask = error_df['Contract Number'] == contract
            
            # Get the list of vendors for error message
            vendors = ', '.join(invalid_contracts[contract])
            
            # Add error message
            error_df.loc[contract_mask, 'Error-Multiple Vendors'] = f'Contract has multiple vendors: {vendors}'
            error_df.loc[contract_mask, 'Has Error'] = True
    
    return error_df


def check_duplicates(error_df, duplicate_mode):
    """Check for potential duplicates based on specified mode"""
    if duplicate_mode == 'default':
        # Use default keys: Contract Number + Reduced Mfg Part Num
        duplicate_keys = ['Contract Number', 'Reduced Mfg Part Num']
    else:
        # Use explicit keys: Contract Number + Mfg Part Num + UOM
        duplicate_keys = ['Contract Number', 'Mfg Part Num', 'UOM']
    
    # Group by duplicate keys and find items that appear more than once
    error_df['Dup Count'] = error_df.groupby(duplicate_keys)['File Row'].transform('count')
    duplicates = error_df['Dup Count'] > 1
    error_df.loc[duplicates, 'Warning-Potential Duplicates'] = 'Potential Duplicates'
    error_df.loc[duplicates, 'Has Error'] = True
    
    # Store duplicate info
    duplicate_info = {
        'mode': duplicate_mode,
        'keys': duplicate_keys,
    }
    
    return error_df, duplicate_info, duplicate_keys


def finalize_validation(error_df, result_df):
    """Drop temporary columns and check for errors"""
    # Drop temporary columns used for validation
    temp_columns = ['Effective_Date_Dt', 'Expiration_Date_Dt', 
                   'Price_Parsed', 'QOE_Parsed', 'Dup Count']
    
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in temp_columns if col in error_df.columns]
    if columns_to_drop:
        error_df.drop(columns_to_drop, axis=1, inplace=True)

    # fillna
    error_df.fillna('', inplace=True)
    result_df.fillna('', inplace=True)

    # Check if there are any errors
    has_errors = error_df['Has Error'].any()
    
    return result_df, error_df, has_errors

def validate_file(df, column_mapping, duplicate_mode='default'):
    """
    Main validation function that coordinates all validation steps
    Returns: (result_df, error_df, has_errors)
    """
    # Prepare dataframe for validation
    error_df, result_df, missing_fields, required_fields = prepare_dataframe(df, column_mapping)
    
    # Check for missing required fields in column mapping
    if missing_fields:
        return None, None, f"Required field(s) {', '.join(missing_fields)} not found"
    
    # Run validation steps
    error_df = validate_required_fields(error_df, required_fields)
    error_df = validate_dates(error_df)
    error_df = validate_prices(error_df)
    error_df = validate_qoe(error_df)
    error_df = validate_uom(error_df)
    error_df = validate_uom_qoe_compatibility(error_df)
    error_df = validate_contract_vendor_relationship(error_df)
    error_df, duplicate_info, duplicate_keys = check_duplicates(error_df, duplicate_mode)
    
    # Finalize and return results
    return finalize_validation(error_df, result_df)

def save_error_file(error_df, user_id, original_filename):
    """Save error file with user-specific filename and return the filename"""
    # Create user-specific directory
    user_dir = os.path.join('temp_files', f'user_{user_id}')
    os.makedirs(os.path.join(current_app.root_path, user_dir), exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"error_report_{timestamp}_{secure_filename(original_filename)}"
    
    # Full path for saving
    file_path = os.path.join(user_dir, filename)
    
    # Save the error file
    error_df.to_excel(os.path.join(current_app.root_path, file_path), index=False)
    
    return file_path

def get_db_connection():
    """Get a connection from the SQLAlchemy pool"""
    try:
        # Get the engine from the Flask app
        engine = current_app.config['DB_ENGINE']
        # Get a connection from the pool
        conn = engine.connect().connection
        return conn
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return None

@contextmanager
def db_transaction():
    """Context manager for database transactions"""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def create_temp_table(table_name, df, conn):
    """Create a temporary table in the database for the uploaded file data
    
    Args:
        table_name: The name to give the temp table
        df: DataFrame containing the data
        conn: Active database connection
    
    Returns:
        tuple: (success boolean, message string)
    """
    try:
        cursor = conn.cursor()
        
        # Drop the table if it already exists
        cursor.execute(f"IF OBJECT_ID('{table_name}', 'U') IS NOT NULL DROP TABLE {table_name}")
        
        # Create the temp table with appropriate columns
        create_table_sql = f"""
        CREATE TABLE {table_name} (
            Mfg_Part_Num VARCHAR(255) NOT NULL,
            Vendor_Part_Num VARCHAR(255),
            Buyer_Part_Num VARCHAR(255),
            Description NVARCHAR(MAX) NOT NULL,
            Contract_Price MONEY NOT NULL,
            UOM VARCHAR(50) NOT NULL,
            QOE INT NOT NULL,
            Effective_Date DATE NOT NULL,
            Expiration_Date DATE NOT NULL,
            Contract_Number VARCHAR(100) NOT NULL,
            ERP_Vendor_ID VARCHAR(20) NOT NULL,
            Reduced_Mfg_Part_Num VARCHAR(255),
            File_Row INT,
            PRIMARY KEY (Mfg_Part_Num, Contract_Number, UOM)
        )
        """
        cursor.execute(create_table_sql)
        
        # Define SQL table columns
        columns = ['Mfg_Part_Num', 'Vendor_Part_Num', 'Buyer_Part_Num', 'Description', 
                'Contract_Price', 'UOM', 'QOE', 'Effective_Date', 'Expiration_Date', 
                'Contract_Number', 'ERP_Vendor_ID', 'Reduced_Mfg_Part_Num', 'File_Row']
        
        # columns we want from our input DataFrame
        pre_checked_columns = ['Mfg Part Num', 'Vendor Part Num', 'Buyer Part Num', 'Description',
                               'Contract Price', 'UOM', 'QOE', 'Effective Date', 'Expiration Date',
                               'Contract Number', 'ERP Vendor ID', 'Reduced Mfg Part Num', 'File Row']
        df = df[pre_checked_columns].copy()
        
        
        # Map from DataFrame columns to SQL table columns
        insert_df = pd.DataFrame(index=df.index)
        for col in columns:
            if col in df.columns:
                insert_df[col] = df[col]
            else:
                # Handle column name variations
                mapped_col = col.replace('_', ' ')
                if mapped_col in df.columns:
                    insert_df[col] = df[mapped_col]
                elif col == 'Buyer_Part_Num':
                    insert_df[col] = ''
                elif col == 'Reduced_Mfg_Part_Num' and 'Reduced Mfg Part Num' in df.columns:
                    insert_df[col] = df['Reduced Mfg Part Num']
                else:
                    print(f"Missing column in DataFrame: {col}")
                    insert_df[col] = None

        # Batch insert for better performance
        for _, row in insert_df.iterrows():
            row_values = [None if pd.isnull(v) or v == '' else v for v in row.values]
            placeholders = ','.join(['?' for _ in row_values])
            column_names = ','.join(columns)
            insert_sql = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
            cursor.execute(insert_sql, row_values)
        
        return True, table_name
    
    except Exception as e:
        print(f"Database error: {str(e)}")
        return False, str(e)
    

def find_duplicates_with_ccx(temp_table, conn):
    """Find potential duplicates between the temp table and CCX database
        
        Args:
            temp_table: Name of the temporary table
            conn: Active database connection
        
        Returns:
            tuple: (success boolean, error message, result list)
        """
    try:
        cursor = conn.cursor()
        
        # SQL query to find potential duplicates
        query = f"""
        SELECT 
            ccx.CONTRACT_NUMBER AS contract_number_ccx,
            ccx.CONTRACT_DESCRIPTION AS contract_description_ccx,
            ccx.CONTRACT_OWNER AS contract_owner_ccx,
            ccx.SOURCE_CONTRACT_TYPE AS source_contract_type_ccx,
            ccx.REDUCED_MANUFACTURER_PART_NUMBER AS reduced_mfg_part_num_ccx,
            ccx.MANUFACTURER_NAME AS manufacturer_name_ccx,
            ccx.MANUFACTURER_PART_NUMBER AS mfg_part_num_ccx,
            ccx.UOM AS uom_ccx,
            ccx.QOE AS qoe_ccx,
            ccx.PRICE AS price_ccx,
            ccx.ITEM_PRICE_START_DATE AS effective_date_ccx,
            ccx.ITEM_PRICE_END_DATE AS expiration_date_ccx,
            ccx.VENDOR_ERP_NUMBER AS erp_vendor_id_ccx,
            ccx.VENDOR_NAME AS vendor_name_ccx,
            ccx.PART_DESCRIPTION AS description_ccx,
            ccx_count.total_line_count AS total_line_count_ccx,
            temp.Mfg_Part_Num,
            temp.Vendor_Part_Num,
            temp.Buyer_Part_Num,
            temp.Description,
            temp.Contract_Price,
            temp.UOM,
            temp.QOE,
            temp.Effective_Date,
            temp.Expiration_Date,
            temp.Contract_Number,
            temp.ERP_Vendor_ID,
            temp.Reduced_Mfg_Part_Num,
            temp.File_Row,
            CASE WHEN ccx.MANUFACTURER_PART_NUMBER = temp.Mfg_Part_Num THEN 1 ELSE 0 END AS same_mfg_part_num
        FROM 
            (
                SELECT 
                    CONTRACT_NUMBER, 
                    CONTRACT_DESCRIPTION, 
                    CONTRACT_OWNER, 
                    SOURCE_CONTRACT_TYPE,
                    CASE
                        WHEN MANUFACTURER_PART_NUMBER IS NULL OR LTRIM(RTRIM(MANUFACTURER_PART_NUMBER)) = '' THEN NULL
                        ELSE
                            CASE
                                WHEN ISNUMERIC(REPLACE(LTRIM(RTRIM(MANUFACTURER_PART_NUMBER)), '-', '')) = 1
                                THEN CAST(TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(MANUFACTURER_PART_NUMBER)), '-', '')) AS VARCHAR(255))
                                ELSE
                                    REPLACE(LTRIM(RTRIM(MANUFACTURER_PART_NUMBER)), '-', '')
                            END
                    END AS REDUCED_MANUFACTURER_PART_NUMBER,
                    MANUFACTURER_NAME, 
                    MANUFACTURER_PART_NUMBER, 
                    UOM,
                    QOE, 
                    PRICE, 
                    ITEM_PRICE_START_DATE, 
                    ITEM_PRICE_END_DATE,
                    VENDOR_ERP_NUMBER, 
                    VENDOR_NAME, 
                    PART_DESCRIPTION
                FROM 
                    [DM_MONTYNT\\dli2].ccx_dump_validation_stg
                WHERE 
                    IS_SUOM = 'f'
                    AND ITEM_PRICE_START_DATE <= LAST_UPDATE_DATE 
                    AND ITEM_PRICE_END_DATE > LAST_UPDATE_DATE
            ) as ccx
        INNER JOIN 
            {temp_table} as temp
        ON 
            CAST(ccx.REDUCED_MANUFACTURER_PART_NUMBER AS VARCHAR(255)) = CAST(temp.Reduced_Mfg_Part_Num AS VARCHAR(255))
        		INNER JOIN
			(
			 SELECT CONTRACT_NUMBER, Total_Line_Count
			 FROM [DM_MONTYNT\\dli2].ccx_dump_line_count_stg
			 ) [ccx_count]
		ON
			ccx_count.CONTRACT_NUMBER = ccx.CONTRACT_NUMBER
        ORDER BY 
            ccx.CONTRACT_NUMBER, ccx.MANUFACTURER_PART_NUMBER
        """
        
        # Execute the query
        cursor.execute(query)
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
        
        # Group by contract number (CCX side)
        contract_summary = {}
        for item in results:
            contract_num = item['contract_number_ccx']
            if contract_num not in contract_summary:
                contract_summary[contract_num] = {
                    'contract_number': contract_num,
                    'contract_description': item['contract_description_ccx'],
                    'manufacturer_name': item['manufacturer_name_ccx'],
                    'contract_owner': item['contract_owner_ccx'],
                    'total_matches': 0,
                    'exact_matches': 0,
                    'total_line_count_ccx': item['total_line_count_ccx'],
                    'items': []
                }
            
            # Count this item
            contract_summary[contract_num]['total_matches'] += 1
            if item['same_mfg_part_num'] == 1:
                contract_summary[contract_num]['exact_matches'] += 1
            
            # Add this item to the contract's items
            contract_summary[contract_num]['items'].append(item)
        
        # Convert to a list
        contract_list = list(contract_summary.values())
        
        return True, "", contract_list
    
    except Exception as e:
        print(f"Database error in find_duplicates_with_ccx: {str(e)}")
        return False, str(e), None

def calculate_mfn_complexity(mfn):
    """Calculate how unique/complex an MFN string is (0.0-1.0)
    
    Args:
        mfn: Manufacturer part number string
        
    Returns:
        float: Complexity score between 0 and 1
    """
    if not mfn or pd.isna(mfn):
        return 0.0
    
    mfn = str(mfn).strip()
    
    # Length factor (longer is better, max at 12 chars)
    length_score = 0.0 if len(mfn) < 3 else min(len(mfn) / 12.0, 1.0)
    
    # Character diversity (unique chars / total length)
    unique_chars = set(mfn)
    diversity_ratio = len(unique_chars) / max(len(mfn), 1)
    
    # Character type variety (digits, letters, special chars)
    has_digits = any(c.isdigit() for c in mfn)
    has_letters = any(c.isalpha() for c in mfn)
    char_type_score = (has_digits + has_letters) / 2.0
    
    # Combined score with weights
    complexity_score = (
        (length_score * 0.6) +          # 60% from length
        (diversity_ratio * 0.2) +       # 20% from character diversity
        (char_type_score * 0.2)         # 20% from character type variety
    )
    
    return complexity_score


def calculate_mfn_match_score(ccx_mfn, upload_mfn):
    """Calculate manufacturer part number match score with complexity consideration (20% weight)
    
    Args:
        ccx_mfn: Manufacturer part number from CCX
        upload_mfn: Manufacturer part number from upload
        
    Returns:
        tuple: (match_score, complexity_score) both between 0 and 1
    """
    # Calculate complexity of the MFNs
    complexity_score = (calculate_mfn_complexity(ccx_mfn) + calculate_mfn_complexity(upload_mfn))/2
    
    # if cmplexity_score < 0.3, we don't trust the the mfn match, so the final mfn score get to multiplied by 0.5
    # if complexity_score > 0.75, exact match and reduced mfn match are both get * 2
    # if complexity_score > 0.85, exact match and reduced mfn match are both get * 3
    # if it is between 0.3 to 0.7, we will trust the mfn match but not the reduced mfn match
    
    # For exact matches, return perfect score
    if ccx_mfn == upload_mfn:
        if complexity_score > 0.85:
            return 3.0, complexity_score
        if complexity_score > 0.7:
            return 2.0, complexity_score
        elif complexity_score < 0.3:
            return 0.5, complexity_score
        return 1.0, complexity_score
    
    # Normalize strings for comparison
    ccx_norm = str(ccx_mfn).strip().lower()
    upload_norm = str(upload_mfn).strip().lower()
    # Remove non-alphanumeric characters
    ccx_alphanum = ''.join(c for c in ccx_norm if c.isalnum())
    upload_alphanum = ''.join(c for c in upload_norm if c.isalnum())
    
    # reduced mfn match
    if ccx_alphanum == upload_alphanum:
        if complexity_score > 0.85:
            return 3.0, complexity_score  # Perfect match with high complexity
        elif complexity_score > 0.70:
            return 2.0, complexity_score
        if complexity_score < 0.3:
            return 0.5, complexity_score  # Perfect match with low complexity
        else:
            return 0.95, complexity_score
    

    # Check if one is contained in the other (only apply to mfn that are longer than 5 letters)
    if ccx_alphanum in upload_alphanum or upload_alphanum in ccx_alphanum:
        base_score = 0.8
        # further constrains on length
        if len(ccx_alphanum) > 5 and len(upload_alphanum) > 5: 
            adjusted_score = base_score * (0.8 + (0.2 * complexity_score))
            return adjusted_score, complexity_score
    
    # If we get here, use Levenshtein distance or fallback
    try:
        from rapidfuzz.distance import Levenshtein
        max_len = max(len(ccx_alphanum), len(upload_alphanum))
        if max_len == 0:
            return 0.0, complexity_score
            
        distance = Levenshtein.distance(ccx_alphanum, upload_alphanum)
        similarity = 1 - (distance / max_len)
        
        # Apply complexity adjustment to Levenshtein similarity
        # Simple strings need to be more similar to get the same score
        base_score = max(0, min(0.5, similarity))
        
        if complexity_score < 0.3:
            # For very simple strings, be more strict
            adjusted_score = base_score * 0.6
        else:
            # For complex strings, be more lenient
            adjusted_score = base_score * (0.7 + (0.3 * complexity_score))
            
        return adjusted_score, complexity_score
    
    except ImportError:
        # Fallback if rapidfuzz is not available
        common_chars = set(ccx_alphanum) & set(upload_alphanum)
        if not common_chars:
            return 0.0, complexity_score
            
        overlap = len(common_chars) / max(len(set(ccx_alphanum)), len(set(upload_alphanum)))
        base_score = max(0, min(0.5, overlap))
        
        # Similar complexity adjustment
        if complexity_score < 0.3:
            adjusted_score = base_score * 0.5
        else:
            adjusted_score = base_score * (0.7 + (0.3 * complexity_score))
            
        return adjusted_score, complexity_score

def calculate_ea_price_match_score(ccx_price, upload_price, ccx_qoe, upload_qoe):
    """Calculate EA price match score (10% weight)
    
    Args:
        ccx_price: Contract price from CCX
        upload_price: Contract price from upload
        ccx_qoe: Quantity of each from CCX
        upload_qoe: Quantity of each from upload
        
    Returns:
        tuple( float: Score, float: EA price percent diff with direction) 
        Score: between 0 and 1
        EA price percent diff with direction: positive if upload > CCX, negative if upload < CCX
    """
    try:
        # Calculate EA price (Contract Price / QOE)
        ccx_ea_price = float(ccx_price) / float(ccx_qoe)
        upload_ea_price = float(upload_price) / float(upload_qoe)
        
        # Prevent division by zero
        if ccx_ea_price == 0 and upload_ea_price == 0:
            return (1.0, 0.0)  # Both prices are zero
        if ccx_ea_price == 0 or upload_ea_price == 0:
            return (0.0, 0.0)  # One price is zero, the other isn't
        
        # Calculate percentage difference (use ccx as base)
        price_diff = abs(upload_ea_price - ccx_ea_price)
        price_diff_direction = 1 if upload_ea_price > ccx_ea_price else -1
        price_diff_percent = price_diff / ccx_ea_price * 100
        price_diff_percent_with_direction = price_diff_percent * price_diff_direction
        
        # Score based on percentage difference
        if price_diff_percent < 10:
            return (1.0, price_diff_percent_with_direction)
        elif price_diff_percent < 25:
            return (0.95, price_diff_percent_with_direction)
        elif price_diff_percent < 50:
            return (0.75, price_diff_percent_with_direction)
        else:
            return (0.0, price_diff_percent_with_direction)
    except (ValueError, TypeError, ZeroDivisionError):
        # Handle conversion or division errors
        return (0.0, 0.0)

def get_sentence_transformer_model():
    """Get the sentence transformer model from app config or local cache if available"""
    from flask import current_app
    
    # Try to get model from app config first
    if current_app.config.get('TRANSFORMER_MODEL_LOADED', False):
        model = current_app.config.get('TRANSFORMER_MODEL')
        if model:
            return model
    
    # Fall back to module cache if not in app config
    return _MODEL_CACHE.get('transformer_model')


def calculate_description_similarity(ccx_desc, upload_desc, model=None):
    """Calculate description similarity score using transformer models
    
    Args:
        ccx_desc: Description from CCX
        upload_desc: Description from upload
        model: Pre-loaded sentence transformer model (optional)
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Normalize strings
    ccx_norm = str(ccx_desc).strip().lower()
    upload_norm = str(upload_desc).strip().lower()
    
    # Handle empty descriptions
    if not ccx_norm or not upload_norm:
        return 0.0
    
    # Check for exact match first
    if ccx_norm == upload_norm:
        return 1.0
    
    # Import necessary libraries (only imported when needed)
    try:
        import re
        import numpy as np
        from scipy.spatial.distance import cosine

        # Use the passed model or get from app config/cache if not provided
        if model is None:
            model = get_sentence_transformer_model()
            if model is None:
                # Silently fall back to simpler approach rather than raising an error
                raise ImportError("No transformer model available")
                
        def extract_numbers_and_measurements(text):
            """Extract and normalize measurements from text descriptions"""
            # Unit normalization mapping
            unit_mapping = {
                'mm': 'mm', 'millimeter': 'mm', 'millimeters': 'mm', 
                'cm': 'cm', 'centimeter': 'cm', 'centimeters': 'cm',
                'in': 'in', 'inch': 'in', 'inches': 'in',
                'ft': 'ft', 'foot': 'ft', 'feet': 'ft',
                'ml': 'ml', 'milliliter': 'ml', 'milliliters': 'ml',
                'l': 'l', 'liter': 'l', 'liters': 'l',
                'kg': 'kg', 'kilogram': 'kg', 'kilograms': 'kg',
                'g': 'g', 'gram': 'g', 'grams': 'g',
                # Add more unit mappings as needed
            }
            
            normalized_measurements = set()
            
            # 1. Extract measurements with units
            measurement_pattern = r'(\d+\.?\d*)\s*([a-zA-Z]+)'
            for match in re.finditer(measurement_pattern, text):
                value, unit = match.groups()
                # Normalize the number by removing trailing zeros
                value = str(float(value)).rstrip('0').rstrip('.') if '.' in value else value
                
                # Normalize the unit if possible
                unit = unit.lower()
                normalized_unit = unit_mapping.get(unit, unit)
                
                # Create normalized measurement string
                normalized_measurements.add(f"{value}{normalized_unit}")
            
            # 2. Extract dimensions (like 10x20x30)
            dimension_pattern = r'(\d+\.?\d*)\s*[xX]\s*(\d+\.?\d*)(?:\s*[xX]\s*(\d+\.?\d*))?'
            for match in re.finditer(dimension_pattern, text):
                dims = [str(float(d)).rstrip('0').rstrip('.') if '.' in d else d for d in match.groups() if d]
                normalized_measurements.add('x'.join(dims))
            
            # 3. Extract standalone numbers
            number_pattern = r'\b(\d+\.?\d*)\b'
            for match in re.finditer(number_pattern, text):
                value = match.group(1)
                # Normalize by removing trailing zeros
                value = str(float(value)).rstrip('0').rstrip('.') if '.' in value else value
                normalized_measurements.add(value)
            
            return normalized_measurements
            
        # Extract numerical components from both descriptions
        ccx_nums = extract_numbers_and_measurements(ccx_norm)
        upload_nums = extract_numbers_and_measurements(upload_norm)
            
        # Calculate numerical overlap score (Jaccard similarity)
        if ccx_nums or upload_nums:
            intersection = len(ccx_nums.intersection(upload_nums))
            union = len(ccx_nums.union(upload_nums))
            numerical_similarity = ((intersection / union) + 1) if union > 0 else 1
        else:
            numerical_similarity = 1  # Neutral score if no numbers present
            
        # Generate embeddings for semantic similarity
        embeddings = model.encode([ccx_norm, upload_norm])
        
        # Calculate cosine similarity between embeddings
        semantic_similarity = 1 - cosine(embeddings[0], embeddings[1])
        
        # Combine scores with weights (70% semantic, 30% numerical)
        combined_similarity = semantic_similarity if numerical_similarity == 1 else (semantic_similarity * 0.5) + (numerical_similarity * 0.5)

        return float(min(max(combined_similarity, 0.0), 1.0))  # Ensure score is between 0 and 1
    
    except Exception as e:
        # Fallback to simpler approach if dependencies aren't available
        # Tokenize descriptions into words
        print("fallback to simple approach", e)
        ccx_words = set(ccx_norm.split())
        upload_words = set(upload_norm.split())
        
        # Calculate Jaccard similarity
        intersection = len(ccx_words.intersection(upload_words))
        union = len(ccx_words.union(upload_words))
        
        if union == 0:
            return 0.0
        
        return intersection / union

def calculate_confidence_score(item, model = None):
    """Calculate overall confidence score based on weighted factors
    
    Args:
        item: Dictionary containing match data
        
    Returns:
        dict: Updated item with confidence scores
    """
    # Make a copy of the item to avoid modifying the original
    result = item.copy()
    
    # Individual factor scores
    mfn_score, mfn_complexity = calculate_mfn_match_score(item['mfg_part_num_ccx'], item['Mfg_Part_Num'])
    
    # Exact match checks
    uom_score = 1.0 if item['uom_ccx'] == item['UOM'] else 0.0
    qoe_score = 1.0 if str(item['qoe_ccx']) == str(item['QOE']) else 0.0
    
    # Price comparison
    price_score, price_diff_pct = calculate_ea_price_match_score(
        item['price_ccx'], item['Contract_Price'],
        item['qoe_ccx'], item['QOE']
    )
    
    # Description similarity
    desc_score = calculate_description_similarity(item['description_ccx'], item['Description'], model = model)
    
    # Calculate EA prices for display
    try:
        ccx_ea_price = float(item['price_ccx']) / float(item['qoe_ccx'])
        upload_ea_price = float(item['Contract_Price']) / float(item['QOE'])
    except (ValueError, TypeError, ZeroDivisionError):
        ccx_ea_price = None
        upload_ea_price = None
    
    # Weighted score calculation
    weighted_score = min((
        (mfn_score * 0.30) +  # MFN match (30%)
        (uom_score * 0.10) +  # UOM match (10%)
        (qoe_score * 0.10) +  # QOE match (10%)
        (price_score * 0.15) + # EA price match (15%)
        (desc_score * 0.35)   # Description similarity (35%)
    ), 1)
    # print(item['Mfg_Part_Num'], mfn_score, mfn_complexity, uom_score, qoe_score, price_score, desc_score, weighted_score)
    
    # Add scores to result
    result['mfn_score'] = mfn_score
    result['mfn_complexity'] = mfn_complexity  # Add this
    result['uom_score'] = uom_score
    result['qoe_score'] = qoe_score
    result['price_score'] = price_score
    result['price_diff_pct'] = price_diff_pct  # Add this
    result['desc_score'] = desc_score
    result['weighted_score'] = weighted_score
    result['ccx_ea_price'] = ccx_ea_price
    result['upload_ea_price'] = upload_ea_price
    
    # Assign confidence level
    if weighted_score >= 0.8:
        result['confidence_level'] = 'high'
    elif weighted_score >= 0.5:
        result['confidence_level'] = 'medium'
    else:
        result['confidence_level'] = 'low'
    
    # Initialize false positive flag
    result['false_positive'] = False
    
    return result

def process_item_comparisons(contract_items, skip_scoring=False, model = None):
    """Process all items and calculate confidence scores
    
    Args:
        contract_items: List of contract items with CCX and upload data
        skip_scoring: Flag to skip scoring
        
    Returns:
        dict: Items grouped by confidence level
    """
    # If model is provided, use it - no need to load again
    # Otherwise, check if we should load it (if not skip_scoring)
    if not model and not skip_scoring and contract_items:
        try:
            # Try to get the model if available
            from flask import current_app
            if current_app.config.get('TRANSFORMER_MODEL_LOADED', False):
                model = current_app.config.get('TRANSFORMER_MODEL')
                print("Using transformer model from app config in process_item_comparisons")
        except Exception as e:
            print(f"Error getting transformer model: {str(e)}")
    
    scored_items = []
    
    if skip_scoring:
        # If skipping scoring, just return the items as is
        scored_items = contract_items
    else:
        # Calculate confidence scores for each item
        for item in contract_items:
            scored_item = calculate_confidence_score(item, model = model)
            scored_items.append(scored_item)
    
    # Group by confidence level
    result = {
        'high': [],
        'medium': [],
        'low': []
    }
    
    for item in scored_items:
        result[item['confidence_level']].append(item)
    
    # Add summary counts
    result['summary'] = {
        'high': {
            'total': len(result['high']),
            'false_positives': 0
        },
        'medium': {
            'total': len(result['medium']),
            'false_positives': 0
        },
        'low': {
            'total': len(result['low']),
            'false_positives': 0
        },
        'total_items': len(scored_items)
    }
    
    return result
