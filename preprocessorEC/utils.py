import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
from flask import current_app
from werkzeug.utils import secure_filename

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

# Replace the entire validate_file function with these smaller functions

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
