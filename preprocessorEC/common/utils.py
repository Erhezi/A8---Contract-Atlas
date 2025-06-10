# Common utilities for the application
import pip_system_certs.wrapt_requests
import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
import unicodedata
from flask import current_app
from werkzeug.utils import secure_filename
from scipy.spatial.distance import cosine
import networkx as nx
import plotly.graph_objects as go
import json

# Global model cache
_MODEL_CACHE = {}

def make_json_serializable(obj):
    """Convert DataFrames and other non-serializable objects to JSON-serializable format"""
    import pandas as pd
    import numpy as np
    
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj
    

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
    """Simplify manufacturer part number by removing dashes and leading zeros and make things upper cased"""
    if pd.isnull(mfg_part_num) or mfg_part_num.strip() == '':
        return np.nan
    mfg_part_num = mfg_part_num.strip().replace('-', '').strip()
    if mfg_part_num.isdigit():
        return str(int(mfg_part_num))
    return mfg_part_num.upper()


def strip_columns(df):
    """Strip whitespace and other wired stuff from all string columns in the DataFrame"""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    return df


def clean_text_data(text):
    """
    Comprehensive text cleaning function that:
    1. Strips whitespace
    2. Removes invisible/control characters from Excel
    3. Removes special characters like trademarks, emojis, and URL encoded characters
    """
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # Strip whitespace
    text = text.strip()
    
    # Remove invisible characters from Excel
    text = (text
        .replace('\u200b', '')  # Zero width space
        .replace('\u200c', '')  # Zero width non-joiner
        .replace('\u200d', '')  # Zero width joiner
        .replace('\u00a0', ' ')  # Non-breaking space
        .replace('\ufeff', '')   # Byte order mark
    )
    
    # Remove URL encoded characters (like %09, %02)
    text = re.sub(r'%[0-9A-Fa-f]{2}', '', text)
    
    # Remove control characters
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    
    # Remove or replace special characters (trademarks, registered symbols, emojis)
    text = (text
        .replace('™', '')
        .replace('®', '')
        .replace('©', '')
    )
    
    # Try to normalize unicode characters (decompose and remove combining marks)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # Remove any other unusual unicode characters like emojis
    text = ''.join(c for c in text if c.isascii() or c.isalpha() or c.isdigit() or c.isspace() or c in '-_.,;:()[]{}#@!?+=/\\')
    
    # Final strip to remove any whitespace created during cleaning
    return text.strip()

def prepare_dataframe(df, column_mapping):
    """Map columns and prepare dataframe for validation"""
    # Required fields (all required except 'Buyer Part Num')
    required_fields = [
        'Mfg Part Num', 'Vendor Part Num', 'Description', 
        'Contract Price', 'UOM', 'QOE', 'Effective Date', 'Expiration Date', 
        'Contract Number', 'ERP Vendor ID', 'Source Contract Type', 'Intended Action'
    ]
    
    # Create a copy of the dataframe with standard field names
    mapped_df = df.copy()
    
    # Apply comprehensive text cleaning to all string columns
    str_columns = mapped_df.select_dtypes(include=['object']).columns
    for col in str_columns:
        mapped_df[col] = mapped_df[col].apply(clean_text_data)
    
    # Replace empty strings with NaN
    mapped_df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    
    # Strip whitespace from string columns (as a safety measure)
    mapped_df = strip_columns(mapped_df)

    # make upper case for Mfg Part Num, Vendor Part Num, UOM, Description, and Contract Number
    for col in ['Mfg Part Num', 'Vendor Part Num', 'UOM', 'Description', 'Contract Number']:
        if col in mapped_df.columns:
            mapped_df[col] = mapped_df[col].str.upper()
    
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
    
    # make vendor erp id standardized
    mapped_df['ERP Vendor ID'] = mapped_df['ERP Vendor ID'].str.strip().str.upper()
    mapped_df['ERP Vendor ID'] = mapped_df['ERP Vendor ID'].apply(lambda x: x[:7])

    # Create copies for validation and results
    result_df = mapped_df[required_fields[:3] + ['Buyer Part Num'] + required_fields[3:] + ['Reduced Mfg Part Num', 'Original UOM', 'File Row']].copy()
    columns_to_save_to_session = result_df.columns.tolist()
    error_df = result_df.copy()
    
    # Add error columns
    error_df['Error-Missing Field'] = ''
    error_df['Error-Invalid Date'] = ''
    error_df['Error-Invalid Price'] = ''
    error_df['Error-Invalid QOE'] = ''
    error_df['Error-Invalid UOM'] = ''
    error_df['Error-EA QOE NOT 1'] = ''
    error_df['Error-Invalid Vendor'] = '' 
    error_df['Error-Invalid Source Contract Type'] = ''
    error_df['Error-Invalid Intended Action'] = ''
    error_df['Warning-Potential Duplicates'] = ''
    error_df['Has Error'] = False
    
    return error_df, columns_to_save_to_session, missing_fields, required_fields

# add a new validation function for Intended Action
def validate_intended_action(error_df):
    """Validate that Intended Action is either 'Upsert' or 'Expire' (case-insensitive)"""
    # First standardize values (convert to title case)
    error_df['Intended Action'] = error_df['Intended Action'].str.strip()
    
    # Create mask for invalid values
    valid_values = ['upsert', 'expire']
    invalid_mask = ~error_df['Intended Action'].str.lower().isin(valid_values)
    
    # Mark errors
    error_df.loc[invalid_mask, 'Error-Invalid Intended Action'] = 'Intended Action must be Upsert or Expire'
    error_df.loc[invalid_mask, 'Has Error'] = True
    
    # Standardize valid values
    standardize_map = {'upsert': 'Upsert', 
                       'expire': 'Expire'}
    valid_mask = ~invalid_mask
    error_df.loc[valid_mask, 'Intended Action'] = error_df.loc[valid_mask, 'Intended Action'].str.lower().map(standardize_map)

    return error_df


# Add a new validation function for Source Contract Type
def validate_source_contract_type(error_df):
    """Validate that Source Contract Type is either 'GPO' or 'Local' (case-insensitive)"""
    # First standardize values (convert to title case)
    error_df['Source Contract Type'] = error_df['Source Contract Type'].str.strip()
    
    # Create mask for invalid values
    valid_values = ['gpo', 'local']
    invalid_mask = ~error_df['Source Contract Type'].str.lower().isin(valid_values)
    
    # Mark errors
    error_df.loc[invalid_mask, 'Error-Invalid Source Contract Type'] = 'Source Contract Type must be GPO or Local'
    error_df.loc[invalid_mask, 'Has Error'] = True
    
    # Standardize valid values
    standardize_map = {'gpo': 'GPO', 'local': 'Local'}
    valid_mask = ~invalid_mask
    error_df.loc[valid_mask, 'Source Contract Type'] = error_df.loc[valid_mask, 'Source Contract Type'].str.lower().map(standardize_map)

    # per contract number should only have one type of source contract type
    contract_types = error_df.groupby('Contract Number')['Source Contract Type'].unique()
    invalid_contracts = contract_types[contract_types.apply(len) > 1]
    if len(invalid_contracts) > 0:
        # For each invalid contract, mark all its rows
        for contract in invalid_contracts.index:
            # Create mask for all rows with this contract
            contract_mask = error_df['Contract Number'] == contract
            
            # Add error message
            error_df.loc[contract_mask, 'Error-Invalid Source Contract Type'] = f'Contract has multiple source types'
            error_df.loc[contract_mask, 'Has Error'] = True
  
    return error_df

def validate_required_fields(error_df, required_fields, duplicate_mode):
    """Check that all required fields have values"""
    forgiven_fields = []
    if duplicate_mode == 'distributor':
        forgiven_fields = ['Vendor Part Num']

    for field in set(required_fields).difference(set(forgiven_fields)):
        missing_mask = error_df[field].isna() | (error_df[field].str.strip() == '')
        error_df.loc[missing_mask, 'Error-Missing Field'] = 'Missing Data'
        error_df.loc[missing_mask, 'Has Error'] = True
    return error_df

def parse_date_safely(date_str):
    if pd.isna(date_str) or str(date_str).strip() == '':
        return pd.NaT
        
    date_str = str(date_str).strip()
    
    # Try multiple formats
    for fmt in [None, '%m/%d/%Y', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S']:
        try:
            if fmt is None:
                return pd.to_datetime(date_str)
            else:
                return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    return pd.NaT

def validate_dates(error_df):
    """Validate date fields format and logic"""
    today_dt = datetime.now().date()

    # Convert date strings to datetime objects
    error_df['Effective_Date_Dt'] = error_df['Effective Date'].apply(parse_date_safely)
    error_df['Expiration_Date_Dt'] = error_df['Expiration Date'].apply(parse_date_safely)

    # Only update strings for valid conversions
    error_df.loc[:, 'Effective Date'] = error_df['Effective_Date_Dt'].dt.strftime('%Y-%m-%d')
    error_df.loc[:, 'Expiration Date'] = error_df['Expiration_Date_Dt'].dt.strftime('%Y-%m-%d')

    # Mark rows with unparseable dates
    empty_date_mask = (error_df['Effective_Date_Dt'].isna() | error_df['Expiration_Date_Dt'].isna())
    error_df.loc[empty_date_mask, 'Error-Invalid Date'] = 'Invalid Date format'
    error_df.loc[empty_date_mask, 'Has Error'] = True

    # Validate expiration date > effective date
    # and expiration date >= today
    invalid_exp_mask = ~empty_date_mask & (
        (error_df['Expiration_Date_Dt'].dt.date < today_dt) | 
        (error_df['Expiration_Date_Dt'] <= error_df['Effective_Date_Dt'])
    )
    error_df.loc[invalid_exp_mask, 'Error-Invalid Date'] = 'Expiration Date must be > Effective Date and >= today'
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
    """Validate that each Contract Number get associated to one ERP Vendor ID,
    deprecated 2025-05-12, as this is allowed and observed in INFOR contract"""
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


def validate_vendor_id(error_df, valid_vids = None):
    """Validate Vendor ID is legitimate"""
    # validate if vendor id is a 7 digit number
    invalid_vid_mask = ~error_df['ERP Vendor ID'].str.match(r'^\d{7}$')
    if valid_vids is not None:
        # Check if vendor ID is in the list of valid vendor IDs
        invalid_vid_mask |= ~error_df['ERP Vendor ID'].isin(valid_vids)       
            
    # Add error message
    error_df.loc[invalid_vid_mask, 'Error-Invalid Vendor'] = 'vendor ID is not Valid'
    error_df.loc[invalid_vid_mask, 'Has Error'] = True
    
    return error_df


def check_duplicates(error_df, duplicate_mode):
    """Check for potential duplicates based on specified mode"""
    if duplicate_mode == 'default':
        # Use default keys: Reduced Mfg Part Num
        duplicate_keys = ['Reduced Mfg Part Num']
    elif duplicate_mode == 'distributor':
        # Use distributor keys: ERP Vendor ID + Mfg Part Num + UOM + Contract Number
        duplicate_keys = ['ERP Vendor ID', 'Mfg Part Num', 'UOM', 'Contract Number']
    else:
        # Use explicit keys: Mfg Part Num + UOM
        duplicate_keys = ['Mfg Part Num', 'UOM']
    
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


def finalize_validation(error_df, columns_to_save_to_session):
    """Drop temporary columns and check for errors"""
    # take price parsed and qoe parsed and save to original
    error_df['Contract Price'] = error_df['Price_Parsed']
    error_df['QOE'] = error_df['QOE_Parsed']

    # Drop temporary columns used for validation
    temp_columns = ['Effective_Date_Dt', 'Expiration_Date_Dt', 
                   'Price_Parsed', 'QOE_Parsed', 'Dup Count']
    
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in temp_columns if col in error_df.columns]
    if columns_to_drop:
        error_df.drop(columns_to_drop, axis=1, inplace=True)

    error_df['Buyer Part Num'] = error_df['Buyer Part Num'].fillna('')
    error_df['Vendor Part Num'] = error_df['Vendor Part Num'].fillna('')

    # Check if there are any errors
    has_errors = error_df['Has Error'].any()

    result_df = error_df[columns_to_save_to_session].copy()
    
    return result_df, error_df, has_errors

def validate_file(df, column_mapping, valid_vids = None, duplicate_mode='default'):
    """
    Main validation function that coordinates all validation steps
    Returns: (result_df, error_df, has_errors)
    """
    # Prepare dataframe for validation
    error_df, columns_to_save_to_session, missing_fields, required_fields = prepare_dataframe(df, column_mapping)
    
    # Check for missing required fields in column mapping
    if missing_fields:
        return None, None, f"Required field(s) {', '.join(missing_fields)} not found"
    
    # Run validation steps
    error_df = validate_required_fields(error_df, required_fields, duplicate_mode)
    error_df = validate_source_contract_type(error_df)  # Add new validation step
    error_df = validate_intended_action(error_df)  # Add new validation step
    error_df = validate_dates(error_df)
    error_df = validate_prices(error_df)
    error_df = validate_qoe(error_df)
    error_df = validate_uom(error_df)
    error_df = validate_uom_qoe_compatibility(error_df)
    error_df = validate_vendor_id(error_df, valid_vids = valid_vids)
    error_df, duplicate_info, duplicate_keys = check_duplicates(error_df, duplicate_mode)
    
    # Finalize and return results
    return finalize_validation(error_df, columns_to_save_to_session)

def save_error_file(error_df, user_id, original_filename):
    """Save error file with user-specific filename and return the filename"""
    # Create user-specific directory
    user_dir = os.path.join(current_app.root_path, 'temp_files', f'user_{user_id}')
    os.makedirs(os.path.join(current_app.root_path, user_dir), exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"error_report_{timestamp}_{secure_filename(original_filename)}"
    
    # Full path for saving
    file_path = os.path.join(user_dir, filename)
    
    # Save the error file
    error_df.to_excel(os.path.join(current_app.root_path, file_path), index=False)
    
    return file_path

# Confidence calculation functions
def calculate_mfn_complexity(mfn):
    """Calculate how unique/complex an MFN string is (0.0-1.0)"""
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
    """Calculate manufacturer part number match score with complexity consideration"""
    # Calculate complexity of the MFNs
    complexity_score = (calculate_mfn_complexity(ccx_mfn) + calculate_mfn_complexity(upload_mfn))/2
    
    # For exact matches, return perfect score with complexity factor
    if ccx_mfn == upload_mfn:
        if complexity_score > 0.85:
            return 3.0, complexity_score
        if complexity_score > 0.70:
            return 2.0, complexity_score
        elif complexity_score < 0.30:
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
            return 2.5, complexity_score  # Perfect match with high complexity
        elif complexity_score > 0.70:
            return 1.5, complexity_score
        if complexity_score < 0.30:
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
    """Calculate EA price match score"""
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
        elif price_diff_percent < 20:
            return (0.95, price_diff_percent_with_direction)
        elif price_diff_percent < 45:
            return (0.75, price_diff_percent_with_direction)
        else:
            return (0.0, price_diff_percent_with_direction)
    except (ValueError, TypeError, ZeroDivisionError):
        # Handle conversion or division errors
        return (0.0, 0.0)

def calculate_description_similarity(ccx_desc, upload_desc, model=None):
    """Calculate description similarity score using transformer models"""
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
        from .model_loader import get_sentence_transformer_model

        # Use the passed model or get from app config/cache if not provided
        if model is None:
            model = get_sentence_transformer_model()
            if model is None:
                # Silently fall back to simpler approach rather than raising an error
                raise ImportError("No transformer model available")
        
        # Add back the measurement extraction function
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
        combined_similarity = semantic_similarity if numerical_similarity == 1 else (semantic_similarity * 0.7) + (numerical_similarity * 0.3)

        return float(min(max(combined_similarity, 0.0), 1.0))  # Ensure score is between 0 and 1
    
    except Exception as e:
        # Fallback to simpler approach if dependencies aren't available
        # Tokenize descriptions into words
        print("Fallback to simple approach:", e)
        ccx_words = set(ccx_norm.split())
        upload_words = set(upload_norm.split())
        
        # Calculate Jaccard similarity
        intersection = len(ccx_words.intersection(upload_words))
        union = len(ccx_words.union(upload_words))
        
        if union == 0:
            return 0.0
        
        return intersection / union

def calculate_confidence_score(item, model=None, apply_to_step=2):
    """Calculate overall confidence score based on weighted factors
    apply_to_step: 2 apply to step 2, 4 apply to step 4"""
    # Make a copy of the item to avoid modifying the original
    apply_to_dict = {
                    2: {
                            'mpn_a': 'mfg_part_num_ccx', 'mpn_upload': 'Mfg_Part_Num',
                            'uom_a': 'uom_ccx', 'uom_upload': 'UOM',
                            'qoe_a': 'qoe_ccx', 'qoe_upload': 'QOE',
                            'price_a': 'price_ccx', 'price_upload': 'Contract_Price',
                            'desc_a': 'description_ccx', 'desc_upload': 'Description',
                            'ea_price_a': 'ccx_ea_price', 'ea_price_upload': 'upload_ea_price'
                        },
                    4: {
                            'mpn_a': 'mfg_part_num_infor', 'mpn_upload': 'Mfg_Part_Num',
                            'uom_a': 'uom_infor', 'uom_upload': 'UOM',
                            'qoe_a': 'qoe_infor', 'qoe_upload': 'QOE',
                            'price_a': 'price_infor', 'price_upload': 'Contract_Price',
                            'desc_a': 'description_infor', 'desc_upload': 'Description',
                            'ea_price_a': 'infor_ea_price', 'ea_price_upload': 'upload_ea_price'
                        }
                    }

    result = item.copy()
    
    # Individual factor scores
    mfn_score, mfn_complexity = calculate_mfn_match_score(item[apply_to_dict[apply_to_step]['mpn_a']], 
                                                          item[apply_to_dict[apply_to_step]['mpn_upload']])
    
    # Exact match checks
    uom_score = 1.0 if item[apply_to_dict[apply_to_step]['uom_a']] == item[apply_to_dict[apply_to_step]['uom_upload']] else 0.0
    qoe_score = 1.0 if str(item[apply_to_dict[apply_to_step]['qoe_a']]).strip() == str(item[apply_to_dict[apply_to_step]['qoe_upload']]).strip() else 0.0
    
    # Price comparison
    price_score, price_diff_pct = calculate_ea_price_match_score(
        item[apply_to_dict[apply_to_step]['price_a']], item[apply_to_dict[apply_to_step]['price_upload']],
        item[apply_to_dict[apply_to_step]['qoe_a']], item[apply_to_dict[apply_to_step]['qoe_upload']])
    
    # Description similarity
    desc_score = calculate_description_similarity(item[apply_to_dict[apply_to_step]['desc_a']], 
                                                  item[apply_to_dict[apply_to_step]['desc_upload']], model=model)
    
    # Calculate EA prices for display
    try:
        ccx_ea_price = float(item[apply_to_dict[apply_to_step]['price_a']]) / float(item[apply_to_dict[apply_to_step]['qoe_a']])
        upload_ea_price = float(item[apply_to_dict[apply_to_step]['price_upload']]) / float(item[apply_to_dict[apply_to_step]['qoe_upload']])
    except (ValueError, TypeError, ZeroDivisionError):
        ccx_ea_price = None
        upload_ea_price = None
    
    # Weighted score calculation - this is already fine-tuned, don't touch my weights
    if desc_score > 0.4:
        weighted_score = min((
            (mfn_score * 0.40) +  # MFN match (40%)
            (uom_score * 0.10) +  # UOM match (10%)
            (qoe_score * 0.05) +  # QOE match (5%)
            (price_score * 0.15) + # EA price match (15%)
            (desc_score * 0.30)   # Description similarity (30%)
        ), 1)
    else:
        weighted_score = min((
            (mfn_score * 0.20) +  # MFN match (20%)
            (uom_score * 0.10) +  # UOM match (10%)
            (qoe_score * 0.05) +  # QOE match (5%)
            (price_score * 0.15) + # EA price match (15%)
            (desc_score * 0.50)   # Description similarity (50%)
        ), 1)
    # print(item['mfg_part_num_ccx'], item['Mfg_Part_Num'], mfn_score, mfn_complexity, uom_score, qoe_score, price_score, price_diff_pct, desc_score, weighted_score)  # Debug log
    
    # Add scores to result
    result['mfn_score'] = mfn_score
    result['mfn_complexity'] = mfn_complexity
    result['uom_score'] = uom_score
    result['qoe_score'] = qoe_score
    result['price_score'] = price_score
    result['price_diff_pct'] = price_diff_pct
    result['desc_score'] = desc_score
    result['weighted_score'] = weighted_score
    result[apply_to_dict[apply_to_step]['ea_price_a']] = ccx_ea_price
    result[apply_to_dict[apply_to_step]['ea_price_upload']] = upload_ea_price
    
    # Assign confidence level
    if weighted_score >= 0.8:
        result['confidence_level'] = 'high'
    elif weighted_score >= 0.6:
        result['confidence_level'] = 'medium'
    else:
        result['confidence_level'] = 'low'
    
    # Initialize false positive flag
    result['false_positive'] = False
    
    return result

def process_item_comparisons(contract_items, skip_scoring=False, model=None, apply_to_step=2):
    """Process all items and calculate confidence scores"""
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
            scored_item = calculate_confidence_score(item, model=model, apply_to_step=apply_to_step)
            scored_items.append(scored_item)
            # print(f"Scored Item: {scored_item}")  # Debug log
    
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


def apply_deduplication_policy(comparison_results, policy, custom_fields=None, sort_directions=None):
    """
    Apply deduplication policy to comparison results
    
    Args:
        comparison_results: Dict with high, medium, low confidence matches
        policy: String indicating dedup policy (custom, newest, prefer_ccx, etc)
        custom_fields: List of field names for custom sorting
        sort_directions: List of sort directions (asc/desc) for custom fields
    
    Returns:
        DataFrame with stacked and sorted data
        Summary dictionary with statistics
    """
    
    # Extract non-false-positive items from all confidence levels
    all_items = []
    for confidence in ['high', 'medium', 'low']:
        items = comparison_results.get(confidence, [])
        true_duplicates = [item for item in items if not item.get('false_positive', False)]
        all_items.extend(true_duplicates)
    
    if not all_items:
        return pd.DataFrame(), {'total_items': 0, 'unique_duplicates': 0}
    
    ccx_data, upload_data = [], []
    for i, item in enumerate(all_items):
    # Create CCX dataframe
        ccx_row = {
            'Source Contract Type': item.get('source_contract_type_ccx', ''),
            'Contract Number': item.get('contract_number_ccx', ''),
            'Total Contract Line Count': item.get('total_line_count_ccx', ''),
            'Reduced Mfg Part Num': item.get('reduced_mfg_part_num_ccx', ''),
            'Mfg Part Num': item.get('mfg_part_num_ccx', ''),
            'Vendor Part Num': item.get('vendor_part_num_ccx', ''),
            'Buyer Part Num': item.get('buyer_part_num_ccx', ''),
            'Description': item.get('description_ccx', ''),
            'UOM': item.get('uom_ccx', ''),
            'QOE': item.get('qoe_ccx', ''),
            'Contract Price': item.get('price_ccx', ''),
            'EA Price': item.get('ccx_ea_price', ''),
            'Effective Date': item.get('effective_date_ccx', ''),
            'Expiration Date': item.get('expiration_date_ccx', ''),
            'ERP Vendor ID': item.get('erp_vendor_id_ccx', ''),
            'Dataset': 'CCX',
            'File Row': item.get('File_Row', ''),  # Group identifier
            'Pair ID': 'tc' + str(i) # Unique identifier for the pair
        }
        ccx_data.append(ccx_row)
    
    # Create upload dataframe
        upload_row = {
            'Source Contract Type': item.get('Source_Contract_Type', ''),
            'Contract Number': item.get('Contract_Number', ''),
            'Total Contract Line Count': item.get('Total_Contract_Line_Count', ''),
            'Reduced Mfg Part Num': item.get('Reduced_Mfg_Part_Num', ''),
            'Mfg Part Num': item.get('Mfg_Part_Num', ''),
            'Vendor Part Num': item.get('Vendor_Part_Num', ''),
            'Buyer Part Num': item.get('Buyer_Part_Num', ''),
            'Description': item.get('Description', ''),
            'UOM': item.get('UOM', ''),
            'QOE': item.get('QOE', ''),
            'Contract Price': item.get('Contract_Price', ''),
            'EA Price': item.get('upload_ea_price', ''),
            'Effective Date': item.get('Effective_Date', ''),
            'Expiration Date': item.get('Expiration_Date', ''),
            'ERP Vendor ID': item.get('ERP_Vendor_ID', ''),
            'Dataset': 'TP',
            'File Row': item.get('File_Row', ''),  # Group identifier
            'Pair ID': 'tc' + str(i) # Unique identifier for the pair
        }
        upload_data.append(upload_row)
    
    # Stack dataframes
    ccx_df = pd.DataFrame(ccx_data) if ccx_data else pd.DataFrame()
    upload_df = pd.DataFrame(upload_data) if upload_data else pd.DataFrame()
    
    if ccx_df.empty and upload_df.empty:
        return pd.DataFrame(), {'total_items': 0, 'unique_duplicates': 0}
    
    # stack the two dataframes
    stacked_df = pd.concat([ccx_df, upload_df], ignore_index=True)
    # make sure the staked_df column show date as YYYY-MM-DD
    stacked_df['Effective Date'] = pd.to_datetime(stacked_df['Effective Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    stacked_df['Expiration Date'] = pd.to_datetime(stacked_df['Expiration Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    # make Contract Price and EA Price in stacked_df show as numeric
    stacked_df['Contract Price'] = pd.to_numeric(stacked_df['Contract Price'], errors='coerce')
    stacked_df['EA Price'] = pd.to_numeric(stacked_df['EA Price'], errors='coerce')
    # make QOE in stacked_df as integer
    stacked_df['QOE'] = pd.to_numeric(stacked_df['QOE'], errors='coerce').astype('Int64')
    # make sure the join key columns are in the same format
    stacked_df['Mfg Part Num'] = stacked_df['Mfg Part Num'].astype(str).str.strip().str.upper()
    stacked_df['Contract Number'] = stacked_df['Contract Number'].astype(str).str.strip().str.upper()
    stacked_df['File Row'] = stacked_df['File Row'].astype(int) 

    # make sure vendor part number and buyer part number are kept as string nan are filled as empty string
    stacked_df['Vendor Part Num'] = stacked_df['Vendor Part Num'].fillna('').str.upper().str.strip()
    stacked_df['Buyer Part Num'] = stacked_df['Buyer Part Num'].fillna('').str.upper().str.strip()

    # dedup the stacked_df so the TP copy only appear once
    # this requires that if the input file (upload TP file) contains multiple contract and if the contract numbers
    # were unknown, they will need assign a unique place holder number to each of the contracts
    stacked_df = stacked_df.drop_duplicates(subset=['File Row', 'Dataset', 'Contract Number'], keep='first')
   
    # # temporarily write out the stacked_df for debugging, store the file to temp_files folder
    # temp_file_path = os.path.join(current_app.root_path, 'temp_files', 'stacked_df_debug.xlsx')
    # stacked_df.to_excel(temp_file_path, index=False)
    # print(stacked_df.columns) # Debug log
    
    # Apply sorting based on policy
    sorted_df = stacked_df.copy()
    
    if (policy == 'custom' or policy == 'manual') and custom_fields:
        # Convert directions to boolean (True for ascending, False for descending)
        ascending = [direction.lower() != 'desc' for direction in sort_directions]
        try:
            sorted_df = stacked_df.sort_values(
                by=custom_fields,
                ascending=ascending
            )
        except Exception as e:
            print(f"Error sorting with custom fields: {e}")
    
    elif policy == 'keep_latest':
        # Sort by Dataset (TP first) and then by dates (newest first)
        sorted_df = stacked_df.sort_values(
            by=['Dataset', 'Expiration Date', 'Effective Date'],
            ascending=[False, False, False]  # 'TP' comes after 'CCX' alphabetically, so we use False to put TP first
        )
    
    elif policy == 'keep_lowest_price':
        # Convert to numeric and sort by price
        sorted_df['EA Price'] = pd.to_numeric(sorted_df['EA Price'], errors='coerce')
        sorted_df = stacked_df.sort_values(
            by=['EA Price', 'Dataset'],
            ascending=[True, False]  # 'TP' comes after 'CCX' alphabetically, so we use False to put TP first
        )

        pass
    
    # Assign rank to each record within its File Row group
    sorted_df['Rank'] = sorted_df.groupby('File Row').cumcount() + 1
    
    if 'Rank' in sorted_df.columns and not sorted_df.empty:
        sorted_df['Keep'] = sorted_df['Rank'] == 1
    else:
        sorted_df['Keep'] = False

    # Generate summary statistics
    results_summary = {
        'total_items': len(sorted_df),
        'unique_duplicates': sorted_df['File Row'].nunique(),
        'kept_ccx': len(sorted_df[(sorted_df['Rank'] == 1) & (sorted_df['Dataset'] == 'CCX')]),
        'kept_uploaded': len(sorted_df[(sorted_df['Rank'] == 1) & (sorted_df['Dataset'] == 'TP')]),
        'duplicates_removed': len(sorted_df) - sorted_df['File Row'].nunique()
    }
    
    return sorted_df, results_summary


def three_way_contract_line_matching(comparison_results, 
                                     infor_cl_match_results,
                                     excluded_contracts):
    """
    Perform three-way contract line matching between CCX, TP, and Infor CL.
    
    Args:
        comparison_results: ccx and upload data with false positive label
        infor_cl_match_results: List of dictionaries containing Infor CL match results
        excluded_contracts: List of excluded contracts from CCX
    
    Returns:
        DataFrame with three-way matched results (one with label result directly, one for showing (dup removed based on contract))
        False Positive: True/False or None
        Need Review: Yes/No
    """
    
    # parse comparison results 
    all_items = []
    for confidence in ['high', 'medium', 'low']:
        items = comparison_results.get(confidence, [])
        for item in items:
            all_items.append(item)
    
    ccx_data = []
    if not all_items:
        print("No duplicates found in CCX with current upload data")
    else:
        for i, item in enumerate(all_items):
        # Create CCX dataframe
            ccx_row = {
                'Source Contract Type': item.get('source_contract_type_ccx', ''),
                'Contract Number': item.get('contract_number_ccx', ''),
                'Reduced Mfg Part Num': item.get('reduced_mfg_part_num_ccx', ''),
                'Mfg Part Num': item.get('mfg_part_num_ccx', ''),
                'Vendor Part Num': item.get('vendor_part_num_ccx', ''),
                'Buyer Part Num': item.get('buyer_part_num_ccx', ''),
                'Description': item.get('description_ccx', ''),
                'UOM': item.get('uom_ccx', ''),
                'QOE': item.get('qoe_ccx', ''),
                'Contract Price': item.get('price_ccx', ''),
                'EA Price': item.get('ccx_ea_price', ''),
                'Effective Date': item.get('effective_date_ccx', ''),
                'Expiration Date': item.get('expiration_date_ccx', ''),
                'ERP Vendor ID': item.get('erp_vendor_id_ccx', ''),
                'Dataset': 'CCX',
                'File Row': item.get('File_Row', ''),  # Group identifier
                'Pair ID': 'tc' + str(i), # Unique identifier for the pair
                'False Positive': item.get('false_positive', False)
            }
            ccx_data.append(ccx_row)
    
    ccx_df = pd.DataFrame(ccx_data) if ccx_data else pd.DataFrame()
        
    # get the infor_cl_match_results and make the data looks similar to stacked_data
    all_items = []
    excluded_contracts = [i.upper().strip() for i in excluded_contracts]
    for group in infor_cl_match_results:
        items = group.get('items', [])
        for item in items:
            if item.get('contract_number_infor', '').upper().strip() in excluded_contracts:
                # skip the excluded contracts
                continue
            all_items.append(item)
        
    all_items_df = pd.DataFrame(all_items) if all_items else pd.DataFrame()

    if all_items_df.empty:
        # no infor matches, then we can simply return empty dataframe and skip to item master matching
        return pd.DataFrame()
    
    # compute EA price
    all_items_df['infor_ea_price'] = all_items_df['price_infor'].astype(float) / all_items_df['qoe_infor'].astype(int)
    all_items_df['upload_ea_price'] = all_items_df['Contract_Price'].astype(float) / all_items_df['QOE'].astype(int)

    # make sure the join key columns are in the same format
    all_items_df['mfg_part_num_infor'] = all_items_df['mfg_part_num_infor'].astype(str).str.strip().str.upper()
    all_items_df['contract_number_infor'] = all_items_df['contract_number_infor'].astype(str).str.strip().str.upper()
    all_items_df['File_Row'] = all_items_df['File_Row'].astype(int)

    ccx_df['Mfg Part Num'] = ccx_df['Mfg Part Num'].astype(str).str.strip().str.upper()
    ccx_df['Contract Number'] = ccx_df['Contract Number'].astype(str).str.strip().str.upper()
    ccx_df['File Row'] = ccx_df['File Row'].astype(int)


    if not ccx_df.empty:
        merged_df = pd.merge(
                            all_items_df, 
                            ccx_df[['File Row', 'Mfg Part Num', 'Contract Number', 'False Positive']],
                            left_on = ['File_Row', 'contract_number_infor', 'mfg_part_num_infor'],
                            right_on = ['File Row', 'Contract Number', 'Mfg Part Num'],
                            how = 'left',
                            indicator = True)
        merged_df.loc[:, 'Need Review'] = merged_df['_merge'].apply(lambda x: 'No' if x == 'both' else 'Yes')
        merged_df = merged_df.drop(columns=['_merge', 'File Row', 'Contract Number', 'Mfg Part Num'])
        # Infor data -- same contract can replicate under different vendor or being loaded twice under different Infor contract object ID
        # we may want to drop them in the future, but for now I will simply keep them if they are there, and return two dataframes
        # merged_df.to_excel(os.path.join(current_app.root_path, 'temp_files', 'merged_df_debug.xlsx'), index=False)
        return merged_df
    
    all_items_df['False Positive'] = False
    all_items_df['Need Review'] = 'Yes'

    merged_df = all_items_df.copy()

    return merged_df


def three_way_item_master_matching_compute_similarity(merged_df):
    """
    Compute similarity scores for three-way item master matching.
    
    Args:
        merged_df: DataFrame containing merged data from CCX and Infor CL
    
    Retruns:
        DataFrame with similarity scores and flags for review
    """
    result = {
        'summary': {
            'review_count': 0,
            'no_need_review_count': 0,
            'item_master_count': 0
        },
        'im_catched': []
    }
    
    if merged_df.empty:
        return result
    # just focusing on contract - item, ignore the the fact that on Infor same contract can replicates
    
    merged_to_show_df = merged_df.drop_duplicates(subset = ['File_Row', 'mfg_part_num_infor', 'contract_number_infor'], keep = 'first')
    need_review_df = merged_to_show_df[merged_to_show_df['Need Review'] == 'Yes'].copy()
    no_need_review_df = merged_to_show_df[merged_to_show_df['Need Review'] == 'No'].copy()

    review_count, no_need_review_count = len(need_review_df), len(no_need_review_df)
    im_count = len(set(no_need_review_df[no_need_review_df['item_number_infor'] != '']['File_Row']))
    im_catched = no_need_review_df[no_need_review_df['item_number_infor'] != ''][['File_Row', 'item_number_infor']].drop_duplicates()
    im_catched = im_catched.rename(columns={'item_number_infor': 'ItemNumber'})
    im_catched = im_catched.drop_duplicates(subset = ['File_Row', 'ItemNumber'], keep = 'first')

    # carve out the portion need to run through confidence score calculation
    to_calc_df = merged_to_show_df[merged_to_show_df['Need Review'] == 'Yes'].copy()
    to_calc_df['False Positive'] = False
    item_list = to_calc_df.to_dict(orient='records')
    
    result = process_item_comparisons(item_list, skip_scoring=False, model=None, apply_to_step=4)
    result['summary']['review_count'] = review_count
    result['summary']['no_need_review_count'] = no_need_review_count
    result['summary']['item_master_count'] = im_count
    result['im_catched'] = im_catched.values.tolist() if not im_catched.empty else []
    
    return result


def make_infor_upload_stack(merged_df):
    """
    Create a stacked DataFrame from Infor CL match results.
    
    Args:
        merged_df: List of dictionaries containing Infor CL match results
    
    Returns:
        DataFrame with stacked Infor CL data
    """
     
    infor_cl_data, upload_cl_data = [], []
    p_cnt = 0
    for item in merged_df:
        if item.get('False Positive') == True:
            # skip the false positive
            continue
        infor_row = {
            'Source Contract Type': 'Not Applicable',
            'Contract Number': item.get('contract_number_infor', ''),
            'Total Contract Line Count': item.get('total_line_count_infor', ''),
            'Reduced Mfg Part Num': item.get('reduced_mfg_part_num_infor', ''),
            'Mfg Part Num': item.get('mfg_part_num_infor', ''),
            'Vendor Part Num': item.get('vendor_part_num_infor', ''),
            'Buyer Part Num': item.get('item_number_infor', ''),
            'Description': item.get('description_infor', ''),
            'UOM': item.get('uom_infor', ''),
            'QOE': item.get('qoe_infor', ''),
            'Contract Price': item.get('price_infor', ''),
            'EA Price': '', # need calculation, it will be the price_infor / qoe_infor, will deal with it later
            'Effective Date': item.get('effective_date_infor', ''),
            'Expiration Date': item.get('expiration_date_infor', ''),
            'ERP Vendor ID': item.get('erp_vendor_id_infor', ''),
            'Dataset': 'Infor',
            'File Row': item.get('File_Row', ''),
            'Pair ID': 'ti' + str(p_cnt), # Unique identifier for the pair
            'ItemNumber': item.get('item_number_infor', ''),
            'Contract ERP ID': item.get('erp_contract_id_infor', '') # same contract number sometimes can be put under two different vendor, thus two contract objects
        }
        tp_row = {
            'Source Contract Type': item.get('Source_Contract_Type', ''),
            'Contract Number': item.get('Contract_Number', ''),
            'Total Contract Line Count': item.get('Total_Contract_Line_Count', ''),
            'Reduced Mfg Part Num': item.get('Reduced_Mfg_Part_Num', ''),
            'Mfg Part Num': item.get('Mfg_Part_Num', ''),
            'Vendor Part Num': item.get('Vendor_Part_Num', ''),
            'Buyer Part Num': item.get('Buyer_Part_Num', ''),
            'Description': item.get('Description', ''),
            'UOM': item.get('UOM', ''),
            'QOE': item.get('QOE', ''),
            'Contract Price': item.get('Contract_Price', ''),
            'EA Price': '', # need calculation, will deal with it later
            'Effective Date': item.get('Effective_Date', ''),
            'Expiration Date': item.get('Expiration_Date', ''),
            'ERP Vendor ID': item.get('ERP_Vendor_ID', ''),
            'Dataset': 'TP',
            'File Row': item.get('File_Row', ''),
            'Pair ID': 'ti' + str(p_cnt) # Unique identifier for the pair
        }
        infor_cl_data.append(infor_row)
        upload_cl_data.append(tp_row)
        p_cnt += 1
        
    infor_df = pd.DataFrame(infor_cl_data) if infor_cl_data else pd.DataFrame()
    upload_df = pd.DataFrame(upload_cl_data) if upload_cl_data else pd.DataFrame()

    stacked_df = pd.concat([infor_df, upload_df], ignore_index=True)
    if not stacked_df.empty:
        # make sure the staked_df column show date as YYYY-MM-DD
        stacked_df['Effective Date'] = pd.to_datetime(stacked_df['Effective Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        stacked_df['Expiration Date'] = pd.to_datetime(stacked_df['Expiration Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        # make Contract Price and EA Price in stacked_df show as numeric
        stacked_df['Contract Price'] = pd.to_numeric(stacked_df['Contract Price'], errors='coerce')
        # make QOE in stacked_df as integer
        stacked_df['QOE'] = pd.to_numeric(stacked_df['QOE'], errors='coerce').astype('Int64')
        # calculate EA Price
        stacked_df['EA Price'] = stacked_df['Contract Price'] / stacked_df['QOE']
        # make sure the join key columns are in the same format
        stacked_df['Mfg Part Num'] = stacked_df['Mfg Part Num'].astype(str).str.strip().str.upper()
        stacked_df['Contract Number'] = stacked_df['Contract Number'].astype(str).str.strip().str.upper()
        stacked_df['File Row'] = stacked_df['File Row'].astype(int)

        stacked_df = stacked_df.drop_duplicates(subset=['File Row', 'Dataset', 'Contract Number'], keep='first')

    return stacked_df


def item_catched_in_infor_im_match(items):
    """
    Check if items are caught in Infor IM match.
    
    Args:
        items: List of dictionaries containing item data
    
    Returns:
        list of list [file row, item number]
    """
    
    # Create a DataFrame from the item list
    df = pd.DataFrame(items)
    if df.empty:
        return []
    # Filter for items with 'ItemNumber' not empty and false positive = False
    filtered_df = df[df['item_number_infor'] != ''].copy()
    if 'false positive' in df.columns:
        filtered_df = df[(df['item_number_infor'] != '') & (df['false_positive'] == False)]
    
    if filtered_df.empty:
        return []
    
    im_catched = filtered_df[['File_Row', 'item_number_infor']].drop_duplicates(keep = 'first')
    
    return im_catched.values.tolist() if not im_catched.empty else []


def extract_item_numbers_for_validation(im_catched_infor_cl, im_catched_infor_im):
    """
    Extract unique item numbers from contract line matches and item master matches
    for UOM validation.
    
    Args:
        im_catched_infor_cl: im_catched from Infor contract line matching
        im_caatched_infor_im: im_catched from item master matching
        
    Returns:
        item_numbers: list of unique item numbers to validate
        im_catched_all_df: DataFrame with all item numbers collected using infor_cl and infor_im match in step4
    """
    item_numbers = set()
    im_catched_all = []
    
    # Extract from contract line matches
    if im_catched_infor_cl:
        for file_row, item_number in im_catched_infor_cl:
            item_numbers.add(item_number)
            im_catched_all.append([file_row, item_number])
    
    # Extract from item master matches
    if im_catched_infor_im:
        for file_row, item_number in im_catched_infor_im:
            item_numbers.add(item_number)
            im_catched_all.append([file_row, item_number])
    
    if len(im_catched_all) == 0:
        return [], pd.DataFrame()
    
    # Convert to DataFrame
    im_catched_all_df = pd.DataFrame(im_catched_all, columns=['File Row', 'Item'])
    im_catched_all_df = im_catched_all_df.drop_duplicates(keep='first')
    # indicate the total numbers of item master item matched per file row
    im_catched_all_df['Item'] = im_catched_all_df['Item'].astype(str).str.strip().str.upper()
    im_catched_all_df['File Row'] = im_catched_all_df['File Row'].astype(int)
    im_catched_all_df.loc[:, 'Matched Count'] = im_catched_all_df.groupby('File Row')['Item'].transform('count')
    
    return list(item_numbers), im_catched_all_df

def analyze_uom_qoe_discrepancies(valid_uom, validated_upload, im_catched_all_df):
    """
    Analyze UOM and QOE discrepancies between validated upload and valid UOM.
    
    Args:
        valid_uom: list of dict containing valid UOM data
        validated_upload: list of dict containing validated upload data
    
    Returns:
        analyzed_df - DataFrame with discrepancies and validation results
    """

    # convert to dataframe
    valid_uom_df = pd.DataFrame(valid_uom)
    valid_uom_df.rename(columns={'UOMConversion': 'QOE'}, inplace=True)
    validated_upload_df = pd.DataFrame(validated_upload)
    # we only need some columns from the validated_upload_df
    validated_upload_df = validated_upload_df[['File Row', 
                                               'ERP Vendor ID', 
                                               'Mfg Part Num',
                                               'Vendor Part Num',
                                               'UOM', 
                                               'QOE',
                                               'Description',
                                               'Contract Number']].copy()

    # make sure the join key columns are in the same format
    valid_uom_df['Item'] = valid_uom_df['Item'].astype(str).str.strip().str.upper()
    
    # Merge the two DataFrames on 'File Row' and 'ItemNumber'
    merged_df = im_catched_all_df.merge(
        validated_upload_df,
        on=['File Row'],
        how='left'
    ).merge(
        valid_uom_df,
        on=['Item'],
        how = 'left',
        suffixes=('_upload', '_im')
    )

    # Check for discrepancies in UOM and QOE
    # UOM need to be string and QOE will be int
    for col in ['UOM_im', 'UOM_upload']:
        merged_df[col] = merged_df[col].astype(str).str.strip().str.upper()
    for col in ['QOE_im', 'QOE_upload']:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').astype('Int64')
    
    merged_df['UOM Check'] = merged_df['UOM_im'] == merged_df['UOM_upload']
    merged_df['QOE Check'] = merged_df['QOE_im'] == merged_df['QOE_upload']

    # isolate any file row with a passed check in UOM or QOE
    passed_file_row = set(merged_df[(merged_df['UOM Check'] == True) & (merged_df['QOE Check'] == True)]['File Row'])
    merged_df.loc[:, 'Validation'] = merged_df['File Row'].apply(lambda x: 'Passed' if x in passed_file_row else 'Failed')
    
    # summarize all possible UOM * QOE from valid_uom_df
    valid_uom_df.loc[:, 'UOM and QOE'] = valid_uom_df['UOM'] + '*' + valid_uom_df['QOE'].astype(int).astype(str)
    valid_uom_df.sort_values(by=['Item', 'QOE'], ascending=[True, True], inplace=True)
    valid_uom_df.loc[:, 'All Valid UOM*QOE'] = valid_uom_df.groupby(['Item'])['UOM and QOE'].transform(lambda x: ','.join(x))

    analyzed_df = merged_df.merge(
        valid_uom_df[['Item', 'All Valid UOM*QOE']],
        on=['Item'],
        how='left'
    )

    analyzed_df = analyzed_df[['File Row',  
                               'Mfg Part Num',
                               'Vendor Part Num',
                               'UOM_upload', 
                               'QOE_upload',
                               'Description',
                               'Contract Number',
                               'ERP Vendor ID',
                               'Item',
                               'All Valid UOM*QOE',
                               'ItemDescription',
                               'Validation',
                               'Matched Count']].copy()
    
    analyzed_df.loc[:, 'False Positive'] = False
    analyzed_df = analyzed_df.drop_duplicates(keep = 'first')

    all_pass_flag = False
    if len(analyzed_df) == 0:
        all_pass_flag = True
    if (len(analyzed_df[analyzed_df['Validation'] == 'Failed']) == 0):
        all_pass_flag = True
    
    one_to_many_warning = True
    if analyzed_df['Matched Count'].max() == 1:
        one_to_many_warning = False
    if len(analyzed_df) == 0:
        one_to_many_warning = False

    results = {
        'analyzed_df': analyzed_df.to_dict(orient='records'),
        'false_positive_count': 0,
        'failed_count': len(analyzed_df[analyzed_df['Validation'] == 'Failed']),
        'total_validation_count': len(analyzed_df),
        'all_pass_flag': all_pass_flag,
        'one_to_many_warning': one_to_many_warning
    }
    
    return results

def recompute_uom_qoe_validation_metrics(analyzed_df):
    """
    Recompute UOM and QOE validation metrics.
    
    Args:
        analyzed_df: List of dict with analyzed UOM and QOE data
    Returns:
        results: Dictionary with validation metrics
    """   
    # Convert to DataFrame
    analyzed_df = pd.DataFrame(analyzed_df)

    if analyzed_df.empty:
        return {
            'false_positive_count': 0,
            'failed_count': 0,
            'total_validation_count': 0,
            'all_pass_flag': True,
            'one_to_many_warning': False
        }
    
    # Recompute metrics
    # exclude false positives
    analyzed_df = analyzed_df[analyzed_df['False Positive'] == False].copy()
    false_positive_count = len(analyzed_df[analyzed_df['False Positive'] == True])
    failed_count = len(analyzed_df[analyzed_df['Validation'] == 'Failed'])
    total_validation_count = len(analyzed_df)
    
    all_pass_flag = False
    if (len(analyzed_df[analyzed_df['Validation'] == 'Failed']) == 0):
        all_pass_flag = True
    
    one_to_many_warning = True
    analyzed_df['Matched Count'] = analyzed_df.groupby('File Row')['Item'].transform('count')
    if analyzed_df['Matched Count'].max() == 1:
        one_to_many_warning = False
    if len(analyzed_df) == 0:
        one_to_many_warning = False
    
    results = {
        'false_positive_count': false_positive_count,
        'failed_count': failed_count,
        'total_validation_count': total_validation_count,
        'all_pass_flag': all_pass_flag,
        'one_to_many_warning': one_to_many_warning
    }
    return results


def change_simulation_stage1(validated_df, stacked_df):
    """
    join validated_df and stacked_df to get insight on how we should make changes in system

    Args:
        validated_df: DataFrame with validated data
        stacked_df: DataFrame with stacked data
    
    Returns:
        df_cross: DataFrame to feed into network graph to show the overlapping contract items
    """
    # count line count for validated_df per contract number
    validated_df['Total Contract Line Count'] = validated_df.groupby('Contract Number')['File Row'].transform('count')
    #
    # stacked df to pass the Keep information if same contract number between TP and CCX
    stacked_df['Contract Number'] = stacked_df['Contract Number'].astype(str).str.strip().str.upper()
    
    # merge dataframe to have the indicator information aligned
    df_m = validated_df.merge(stacked_df[stacked_df['Dataset'] == 'CCX'],
                              on = ['File Row'],
                              how = 'left',
                              suffixes = ('_a', '_b'),
                              indicator = True)
    
    # mark for same contract number (make sure they are upper cased and stripped)
    df_m['Contract Number_a'] = df_m['Contract Number_a'].astype(str).str.strip().str.upper()
    df_m['Contract Number_b'] = df_m['Contract Number_b'].fillna('').astype(str).str.strip().str.upper()
    df_m['Same Contract Number'] = df_m['Contract Number_a']== df_m['Contract Number_b']
    df_m['Total Contract Line Count_b'] = df_m['Total Contract Line Count_b'].fillna(0).astype(int)

    # group by contract pairs to get the count of overlappings
    df_cross = df_m.groupby(['_merge',
                             'Contract Number_a', 
                             'Contract Number_b',
                             'Total Contract Line Count_a',
                             'Total Contract Line Count_b'],
                             observed = True).agg({'File Row': 'nunique'}).reset_index()
    df_cross.rename(columns = {'File Row': 'Overlapping Count'}, inplace = True)
    df_cross.to_excel(os.path.join(current_app.root_path, 'temp_files', 'df_cross.xlsx'), index = False) #debug
    
    return df_cross

def actual_action_on_update_row(row, update_action_mode = None):
    fields_to_compare = ['Mfg Part Num', 'Vendor Part Num',
                        'Buyer Part Num', 'Description', 'Contract Price',
                        'UOM', 'QOE', 'Effective Date', 'Expiration Date']
    
    comparison_results = []
    for field in fields_to_compare:
        if field != 'Buyer Part Num':
            field_keep = field + '_keep'
            field_drop = field + '_drop'
            res = row[field_keep] == row[field_drop]
            comparison_results.append('Y' if res == True else 'N')
        else:
            comparison_results.append('x')
    
    quick_check = ''.join(comparison_results)
    
    if quick_check == 'YYxYYYYYY':
        # No change in the fields we care about
        return "No Change", quick_check
    
    # quick check
    quick_check = ''.join(comparison_results)
    if update_action_mode == 'legacy':
        # the field currently is not used but we want to keep it in code to make things clear
        pure_update_fields = ['Description', 'Contract Price']
        # pure_update_fields = ['Description']
        
        # Check if only description changed
        if quick_check in ['YYxNYYYYY', 'YYxYNYYYY', 'YYxNNYYYY']:
        # if quick_check == 'YYxNYYYYY':
            return "Update (New)", quick_check
        
        # Any other field changeds
        return "Expire then Create (Create)", quick_check
    
    elif update_action_mode == 'new':
        expire_then_create_fields = ['Mfg Part Num', 'UOM']
        if quick_check[0] == 'N' or quick_check[5] == 'N':
            # if Mfg Part Num or UOM changed, we need to expire then create
            return "Expire then Create (Create)", quick_check
        else:
            # if only other fields changed, we can update
            return "Update (New)", quick_check
        

def final_data_helper(row, group = 'keep', 
                      actual_action = None,
                      quick_check = None,
                      primary_action = None):
    if group == 'keep':
        return {
            'File Row': row['File Row'],
            'Dataset': row['Dataset_keep'],
            'Contract Number': row['Contract Number_keep'],
            'Mfg Part Num': row['Mfg Part Num_keep'],
            'Vendor Part Num': row['Vendor Part Num_keep'],
            'Buyer Part Num': row['Buyer Part Num_keep'] if row['Buyer Part Num_keep'] != '' else row['Mfg Part Num_drop'],
            'Description': row['Description_keep'],
            'Contract Price': row['Contract Price_keep'],
            'UOM': row['UOM_keep'],
            'QOE': row['QOE_keep'],
            'Effective Date': row['Effective Date_keep'],
            'Expiration Date': row['Expiration Date_keep'],
            'ERP Vendor ID': row['ERP Vendor ID_keep'],
            'Actual Action': actual_action,
            'Quick Check': quick_check,
            'Primary Action': primary_action,
            'Group': 'Keep'
        }
    elif group == 'drop':
        return {
            'File Row': row['File Row'],
            'Dataset': row['Dataset_drop'],
            'Contract Number': row['Contract Number_drop'],
            'Mfg Part Num': row['Mfg Part Num_drop'],
            'Vendor Part Num': row['Vendor Part Num_drop'],
            'Buyer Part Num': row['Buyer Part Num_drop'],
            'Description': row['Description_drop'],
            'Contract Price': row['Contract Price_drop'],
            'UOM': row['UOM_drop'],
            'QOE': row['QOE_drop'],
            'Effective Date': row['Effective Date_drop'],
            'Expiration Date': row['Expiration Date_drop'],
            'ERP Vendor ID': row['ERP Vendor ID_drop'],
            'Actual Action': actual_action,
            'Quick Check': quick_check,
            'Primary Action': primary_action,
            'Group': 'Drop'
        }
    else:
        raise ValueError("Invalid group specified. Use 'keep' or 'drop'.")


def change_simulation_stage2(validated_df, stacked_df, update_action_mode = 'new'):
    upsert_file_row = set(validated_df[validated_df['Intended Action'] == 'Upsert']['File Row'])
    stacked_df['Contract Number'] = stacked_df['Contract Number'].astype(str).str.strip().str.upper()
    keep_df = stacked_df[stacked_df['Keep'] == True].copy()
    drop_df = stacked_df[stacked_df['Keep'] == False].copy()
    
    df_m = keep_df.merge(drop_df,
                         on = ['File Row'],
                         suffixes = ('_keep', '_drop'),
                         how = 'left')
    
    df_m['Intended Action'] = df_m['File Row'].apply(lambda x: 'Upsert' if x in upsert_file_row else 'Expire')
    
    # for df_m matched lines that marked as intention as 'Upsert'
    primary_action = []
    actual_action = []
    quick_check = []
    final_data = []
    update_rows = set()
    expire_rows = set()
    for i, row in df_m.iterrows():
        a_action, q_check = actual_action_on_update_row(row, update_action_mode = update_action_mode) #we can choose different action mode here
        if row['Intended Action'] == 'Upsert':
            if row['Dataset_keep'] == 'TP':
                if row['Contract Number_keep'] == row['Contract Number_drop']:
                    # contract line already exists on CCX, for this row our primary action is to update the existing contract
                    primary_action.append('Update')
                    actual_action.append(a_action)
                    quick_check.append(q_check)
                    update_rows.add(row['File Row']) # if tp row is qualified for update, then we cannot create it later when it has more matches
                    final_data.append(final_data_helper(row, group = 'keep', actual_action = a_action, quick_check = q_check, primary_action = 'Update'))
                    if a_action == 'Expire then Create (Create)':
                        final_data.append(final_data_helper(row, group = 'drop', actual_action = 'Expire then Create (Expire)', quick_check = q_check, primary_action = 'Update'))
                    if a_action == 'Update (New)':
                        final_data.append(final_data_helper(row, group = 'drop', actual_action = 'Update (Existing)', quick_check = q_check, primary_action = 'Update'))
                    if a_action == 'No Change':
                        final_data.append(final_data_helper(row, group = 'drop', actual_action = 'No Change', quick_check = q_check, primary_action = 'No Change'))
                else:
                    # for ccx side, we expire the row
                    primary_action.append('Expire CCX')
                    actual_action.append('Expire')
                    quick_check.append(q_check)
                    final_data.append(final_data_helper(row, group = 'drop', actual_action = 'Expire', quick_check = q_check, primary_action = 'Expire CCX'))
                    # for tp side, we create new row (this can conflict with update row above)
                    final_data.append(final_data_helper(row, group = 'keep', actual_action = 'Create', quick_check = q_check, primary_action = 'Create TP'))
            elif row['Dataset_keep'] == 'CCX':
                if row['Dataset_drop'] == 'TP':
                    # if the drop contract is from TP, then basically we choose to not use the TP version but the CCX version
                    primary_action.append('Mute TP')
                    actual_action.append('Mute')
                    quick_check.append(q_check)
                    final_data.append(final_data_helper(row, group = 'drop', actual_action = 'Mute', quick_check = q_check, primary_action = 'Mute TP'))
                    final_data.append(final_data_helper(row, group = 'keep', actual_action = 'No Change', quick_check = q_check, primary_action = 'No Change'))
                else: 
                    primary_action.append('Expire CCX')
                    actual_action.append('Expire')
                    quick_check.append(q_check)
                    final_data.append(final_data_helper(row, group = 'drop', actual_action = 'Expire', quick_check = q_check, primary_action = 'Expire CCX'))
                    final_data.append(final_data_helper(row, group = 'keep', actual_action = 'No Change', quick_check = q_check, primary_action = 'No Change'))
        
        # when intended action is 'Expire', we simply expire the contract from TP and keep other thing untouched
        elif row['Intended Action'] == 'Expire':
            if row['Dataset_keep'] == 'TP': # drop will always be made on CCX side, we only care if it is paired to our TP, if no, skip
                if row['Dataset_drop'] == 'CCX' and (row['Contract Number_keep'] == row['Contract Number_drop']):
                    primary_action.append('Expire CCX')
                    actual_action.append('Expire')
                    quick_check.append(q_check)
                    expire_rows.add(row['File Row']) # if tp row is qualified for expire, then we cannot mute it later when it has more matches
                    final_data.append(final_data_helper(row, group = 'drop', actual_action = 'Expire', quick_check = q_check, primary_action = 'Expire CCX'))
                    final_data.append(final_data_helper(row, group = 'keep', actual_action = 'Merged', quick_check = q_check, primary_action = 'Merged')) 
                elif row['Dataset_drop'] == 'CCX' and (row['Contract Number_keep'] != row['Contract Number_drop']):
                    primary_action.append('Mute TP')
                    actual_action.append('Mute')
                    quick_check.append(q_check)
                    final_data.append(final_data_helper(row, group = 'keep', actual_action = 'Mute', quick_check = q_check, primary_action = 'Mute TP'))
                    final_data.append(final_data_helper(row, group = 'drop', actual_action = 'No Change', quick_check = q_check, primary_action = 'No Change'))
            elif row['Dataset_keep'] == 'CCX': # drop will always be made on CCX side, we only care if it is paired to our TP, if no, skip
                if row['Dataset_drop'] == 'TP' and (row['Contract Number_keep'] == row['Contract Number_drop']):
                    primary_action.append('Expire CCX')
                    actual_action.append('Expire')
                    quick_check.append(q_check)
                    expire_rows.add(row['File Row']) # if tp row is qualified for expire, then we cannot mute it later when it has more matches
                    final_data.append(final_data_helper(row, group = 'keep', actual_action = 'Expire', quick_check = q_check, primary_action = 'Expire CCX'))
                    final_data.append(final_data_helper(row, group = 'drop', actual_action = 'Merged', quick_check = q_check, primary_action = 'Merged'))
                elif row['Dataset_drop'] == 'TP' and (row['Contract Number_keep'] != row['Contract Number_drop']):
                    primary_action.append('Mute TP')
                    actual_action.append('Mute')
                    quick_check.append(q_check)
                    final_data.append(final_data_helper(row, group = 'drop', actual_action = 'Mute', quick_check = q_check, primary_action = 'Mute TP'))
                    final_data.append(final_data_helper(row, group = 'keep', actual_action = 'No Change', quick_check = q_check, primary_action = 'No Change'))
                else:
                    # it will be nothing, because if keep and drop are both CCX, we don't care.
                    primary_action.append('No Change')
                    actual_action.append('No Change')
                    quick_check.append('xx')
                    final_data.append(final_data_helper(row, group = 'keep', actual_action = 'No Change', quick_check = 'xx', primary_action = 'No Change'))
                    final_data.append(final_data_helper(row, group = 'drop', actual_action = 'No Change', quick_check = 'xx', primary_action = 'No Change'))
    
    df_m['Primary Action'] = primary_action
    df_m['Actual Action'] = actual_action
    df_m['Quick Check'] = quick_check
    df_m.to_excel(os.path.join(current_app.root_path, 'temp_files', 'df_m.xlsx'), index=False) #debug

    data_change_df1 = pd.DataFrame(final_data)

    data_change_df1.loc[:, 'Intended Action'] = data_change_df1['File Row'].apply(lambda x: 'Upsert' if x in upsert_file_row else 'Expire')
    # if update_row has value then we need to solve potential conflict
    if len(update_rows) > 0:
        create_tp_to_remove = data_change_df1[(data_change_df1['Actual Action'] == 'Create') 
                                              & (data_change_df1['File Row'].isin(update_rows))].index
        data_change_df1.drop(index=create_tp_to_remove, inplace=True)
    if len(expire_rows) > 0:
        mute_tp_to_remove = data_change_df1[(data_change_df1['Actual Action'] == 'Mute') 
                                              & (data_change_df1['File Row'].isin(expire_rows))].index
        data_change_df1.drop(index=mute_tp_to_remove, inplace=True)
    
    data_change_df1.drop_duplicates(subset=['File Row', 'Dataset', 'Contract Number', 'Actual Action'], keep='first', inplace=True)

    # merge the net new item from TP to data_change_df
    all_file_row = set(validated_df['File Row'])
    keep_file_row = set(keep_df['File Row'])
    net_new_file_row = (all_file_row.difference(keep_file_row)).intersection(upsert_file_row)
    net_new_df = validated_df[validated_df['File Row'].isin(net_new_file_row)].copy()
    net_new_df['Dataset'] = 'TP'
    net_new_df['Actual Action'] = net_new_df['Intended Action'].apply(lambda x: 'Create' if x == 'Upsert' else 'Mute')
    net_new_df['Quick Check'] = net_new_df['Intended Action'].apply(lambda x: 'x' if x == 'Upsert' else 'xx')  # No quick check for new items
    net_new_df['Primary Action'] = net_new_df['Intended Action'].apply(lambda x: 'Create' if x == 'Upsert' else 'Mute TP')  # New items are created
    net_new_df['Group'] = 'Keep'  # New items are considered as 'Keep'
    data_change_df2 = net_new_df[list(data_change_df1.columns)].copy()

    # combine the two dataframes
    data_change_show_df = pd.concat([data_change_df1, data_change_df2], ignore_index=True)

    # adjust the Effective and Expiration Date for certain operations
    today = pd.to_datetime('today').strftime('%Y-%m-%d')
    tomorrow = (pd.to_datetime('today') + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    for i, row in data_change_show_df.iterrows():
        if row['Actual Action'] == 'Expire' or row['Actual Action'] == 'Expire then Create (Expire)':
            data_change_show_df.at[i, 'Expiration Date'] = today
        elif row['Actual Action'] == 'Expire then Create (Create)':
            data_change_show_df.at[i, 'Effective Date'] = tomorrow

    # data_change_df = data_change_show_df.copy()
    # data_change_df.drop(columns = ['Dataset', 'Quick Check', 'Primary Action', 'Group'], inplace = True)
    
    data_change_show_df.to_excel(os.path.join(current_app.root_path, 'temp_files', 'data_change_show_df.xlsx'), index=False) #debug

    return data_change_show_df


def apply_change(data_change_show_df,
                 validated_df,
                 stacked_df):
    """Apply changes to validated_df and ccx_df isolated from stacked_df,
    return the resulting dataframes."""
    data_change_df = data_change_show_df.copy()
    data_change_df.drop(columns = ['Quick Check', 'Primary Action', 'Group'], inplace = True)

    ccx_df = stacked_df[stacked_df['Dataset'] == 'CCX'].copy()
    
    ccx_change_df = data_change_df[(data_change_df['Dataset'] == 'CCX')].copy()
    
    tp_change_df = data_change_df[(data_change_df['Dataset'] == 'TP')].copy()
                                                                         
    ccx_merge = ccx_df.merge(ccx_change_df,
                             on = ['File Row', 'Contract Number'],
                             how = 'left',
                             suffixes = ('_original', '_change')
    )

    # ccx_merge can have blank change, those are CCX existing record we didn't touch with TP
    ccx_merge_file_rows = set(ccx_merge[~ccx_merge['Actual Action'].isnull()]['File Row'])

    # if actual action is Expire or Create then Expire (Expire), we need to break the file row link
    ccx_merge.loc[:, 'File Row Modified'] = ccx_merge['File Row'].astype(str)
    ccx_merge.loc[ccx_merge['Actual Action'] == 'Expire', 'File Row Modified'] = 'CCX_only'
    ccx_merge.loc[ccx_merge['Actual Action'] == 'Expire then Create (Expire)', 'File Row Modified'] = 'CCX_only'
       
    ccx_merge.to_excel(os.path.join(current_app.root_path, 'temp_files', 'ccx_merge.xlsx'), index=False) #debug
    
    tp_merge = validated_df.merge(tp_change_df,
                                  on = ['File Row', 'Contract Number'],
                                  how = 'left',
                                  suffixes = ('_original', '_change')
    )
    tp_merge.loc[:, 'Actual Action'] = tp_merge.apply(lambda x: 'Merged' 
                                                      if (x['File Row'] in ccx_merge_file_rows and x['Actual Action'] not in (['Create', 'Mute']))
                                                      else x['Actual Action'], axis=1)
    # fill the blank _change columns with _original values
    for col in tp_merge.columns:
        if col.endswith('_change'):
            original_col = col.replace('_change', '_original')
            tp_merge.loc[:, col] = tp_merge[col].fillna(tp_merge[original_col])

    # if actual action is not Create, we need to break the file row link
    tp_merge.loc[:, 'File Row Modified'] = tp_merge['File Row'].astype(str)
    tp_merge.loc[tp_merge['Actual Action'] != 'Create', 'File Row Modified'] = 'TP_only'
    
    tp_merge.to_excel(os.path.join(current_app.root_path, 'temp_files', 'tp_merge.xlsx'), index=False) #debug

    return ccx_merge, tp_merge


def change_simulation_stage3(ccx_merge, tp_merge, data_change_show_df, contract_line_count_df):
    """
    Finalize the changes by applying the changes to CCX and TP dataframes.
    
    Args:
        ccx_merge: DataFrame with CCX changes
        tp_merge: DataFrame with TP changes
        data_change_show_df: DataFrame with changes to be applied
        contract_line_count_df: DataFrame with contract line counts
    -----------
    
    Returns:
        df_cross_new: Dataframe that feed into network graph to replot the contract relations
    """
    action_map = {'No Change': 0,
                'Update (New)': 0,
                'Update (Existing)': 0,
                'Expire': -1,
                'Expire then Create (Create)': 1,
                'Expire then Create (Expire)': -1,
                'Create': 1,
                'Merged': -1, #it means something is deleted from TP (and deletion also find its buddy on CCX)
                'Mute': -1}
    
    # compute the line count changes through different operations
    line_count_cal = data_change_show_df.groupby(['Contract Number', 'Actual Action']).agg({'File Row': 'count'}).unstack(fill_value=0)
    line_count_cal.columns = line_count_cal.columns.droplevel(0)  # Flatten the MultiIndex columns
    line_count_cal = line_count_cal.reset_index()
   
    # Function to calculate delta for each contract
    def calculate_contract_delta(row, action_map):
        delta = 0
        for action, multiplier in action_map.items():
            if action in row.index and not pd.isna(row[action]):
                delta += row[action] * multiplier
        return delta

    def calculate_detailed_changes(row):
        changes = {'Insert': 0, 'Update': 0, 'Delete': 0}
        
        # Insert: Create actions and Expire then Create (Create)
        if 'Create' in row.index:
            changes['Insert'] += row.get('Create', 0)
        if 'Expire then Create (Create)' in row.index:
            changes['Insert'] += row.get('Expire then Create (Create)', 0)
            
        # Update: Update (New) actions
        if 'Update (New)' in row.index:
            changes['Update'] += row.get('Update (New)', 0)
        # if 'Update (Existing)' in row.index:
        #     changes['Update'] += row.get('Update (Existing)', 0)
            
        # Delete: Expire actions and Expire then Create (Expire)
        if 'Expire' in row.index:
            changes['Delete'] += row.get('Expire', 0)
        if 'Expire then Create (Expire)' in row.index:
            changes['Delete'] += row.get('Expire then Create (Expire)', 0)
            
        return changes

    # Calculate deltas for contracts
    if not line_count_cal.empty:
        line_count_cal['Delta'] = line_count_cal.apply(
            lambda row: calculate_contract_delta(row, action_map), axis=1
        )
        detailed_changes = line_count_cal.apply(calculate_detailed_changes, axis=1)
        line_count_cal['Insert_Count'] = detailed_changes.apply(lambda x: x['Insert'])
        line_count_cal['Update_Count'] = detailed_changes.apply(lambda x: x['Update'])
        line_count_cal['Delete_Count'] = detailed_changes.apply(lambda x: x['Delete'])

    # merge line count to current contract line count pulled
    contract_line_count_df['Contract Number'] = contract_line_count_df['Contract Number'].astype(str).str.strip().str.upper()
    line_operations = contract_line_count_df.merge(line_count_cal, on='Contract Number', how='outer')
    line_operations['Total Contract Line Count'] = line_operations['Total Contract Line Count'].fillna(0).astype(int)
    line_operations['Total Contract Line Count (Change Applied)'] = line_operations['Total Contract Line Count'] + line_operations['Delta']

    line_operations.to_excel(os.path.join(current_app.root_path, 'temp_files', 'line_operations.xlsx'), index=False) #debug

    ccx_set = set(ccx_merge['Contract Number'].dropna().astype(str).str.strip().str.upper())
    tp_set = set(tp_merge['Contract Number'].dropna().astype(str).str.strip().str.upper())
    ccx_line_count_cal = line_operations[line_operations['Contract Number'].isin(ccx_set)].copy()
    tp_line_count_cal = line_operations[line_operations['Contract Number'].isin(tp_set)].copy()
    line_count_before_after = line_operations[['Contract Number',
                                                'Total Contract Line Count', 
                                                'Total Contract Line Count (Change Applied)']].copy()


    # limit by Actual Action, then only take the _change portion of them
    ccx_final = ccx_merge[ccx_merge['Actual Action'].isin(['No Change', 
                                                        'Update (New)',
                                                        'Expire then Create (Create)',
                                                        'Expire'])].copy()
    
    tp_final = tp_merge[tp_merge['Actual Action'].isin(['Create'])].copy()

    ccx_change_cols = [col for col in ccx_merge.columns if col.endswith('_change')]
    tp_change_cols = [col for col in tp_merge.columns if col.endswith('_change')]
    ccx_cols_to_keep = ccx_change_cols + ['Actual Action', 
                                          'File Row', 'Contract Number', 
                                          'File Row Modified']
    tp_cols_to_keep = tp_change_cols + ['Actual Action', 
                                        'File Row', 'Contract Number', 
                                        'File Row Modified']
    ccx_final = ccx_final[ccx_cols_to_keep].copy()
    tp_final = tp_final[tp_cols_to_keep].copy()
    # rename the columns to make them consistent
    ccx_final.rename(columns=lambda x: x.replace('_change', ''), inplace=True)
    tp_final.rename(columns=lambda x: x.replace('_change', ''), inplace=True)

    # Merge line count calculations with change details
    merge_cols = ['Contract Number', 'Total Contract Line Count (Change Applied)', 
                      'Insert_Count', 'Update_Count', 'Delete_Count']
    # merge line_count_cal back to ccx_final and tp_final
    ccx_final_lc = ccx_final.merge(line_operations[merge_cols],
                                on='Contract Number',
                                how='left')
    tp_final_lc = tp_final.merge(line_operations[merge_cols],
                                on='Contract Number',
                                how='left')
    
    # fillna for the line count columns
    for col in ['Insert_Count', 'Update_Count', 'Delete_Count']:
        ccx_final_lc[col] = ccx_final_lc[col].fillna(0).astype(int)
        tp_final_lc[col] = tp_final_lc[col].fillna(0).astype(int)
    
    df_m_new = tp_final_lc.merge(ccx_final_lc,
                                      on = ['File Row Modified'],
                                      how = 'outer',
                                      suffixes = ('_a', '_b'),
                                      indicator = True)
    
    df_m_new.to_excel(os.path.join(current_app.root_path, 'temp_files', 'df_m_new.xlsx'), index=False) #debug

    # fillna before groupby
    df_m_new['Contract Number_a'] = df_m_new['Contract Number_a'].fillna('').astype(str).str.strip().str.upper()
    df_m_new['Contract Number_b'] = df_m_new['Contract Number_b'].fillna('').astype(str).str.strip().str.upper()
    # Fill change count columns
    for suffix in ['_a', '_b']:
        for col in ['Insert_Count', 'Update_Count', 'Delete_Count', 'Total Contract Line Count (Change Applied)']:
            df_m_new[f'{col}{suffix}'] = df_m_new[f'{col}{suffix}'].fillna(0).astype(int)

    groupby_cols = ['_merge']
    for suffix in ['_a', '_b']:
        for col in ['Contract Number', 'Total Contract Line Count (Change Applied)',
                    'Insert_Count', 'Update_Count', 'Delete_Count']:
            groupby_cols.append(f'{col}{suffix}')  
    
    df_cross_new = df_m_new.groupby(groupby_cols,
                                    observed = True).agg({'File Row Modified': 'nunique'}).reset_index()
    df_cross_new.rename(columns = {'File Row Modified': 'Deltas'}, inplace = True)
    df_cross_new['Overlapping Count'] = df_cross_new.apply(lambda x: x['Deltas'] if x['_merge'] == 'both' else 0, axis=1)

    df_cross_new.rename(columns = {'Total Contract Line Count (Change Applied)_a': 'Total Contract Line Count_a',
                                   'Total Contract Line Count (Change Applied)_b': 'Total Contract Line Count_b'}, inplace = True)

    df_cross_new.to_excel(os.path.join(current_app.root_path, 'temp_files', 'df_cross_new.xlsx'), index=False) #debug

    return df_cross_new, ccx_line_count_cal, tp_line_count_cal, line_count_before_after

def compute_dataset_changes_df(data_change_show_df):
    if 'Do Not Expire' in data_change_show_df.columns:
        data_change_show_df = data_change_show_df[data_change_show_df['Do Not Expire'] != True].copy()
    # retrun the dataframes to the frontend for display for each change stats card
    ccx_create = data_change_show_df[((data_change_show_df['Primary Action'] == 'Create TP') & (data_change_show_df['Actual Action'] == 'Create')) | 
                                        (data_change_show_df['Actual Action'] == 'Expire then Create (Create)')].copy()
    ccx_update = data_change_show_df[(data_change_show_df['Actual Action'] == 'Update (New)')].copy()
    ccx_expire = data_change_show_df[(data_change_show_df['Actual Action'] == 'Expire') | 
                                        (data_change_show_df['Actual Action'] == 'Expire then Create (Expire)')].copy()
    tp_create = data_change_show_df[((data_change_show_df['Primary Action'] == 'Create') & (data_change_show_df['Actual Action'] == 'Create'))].copy()
    tp_mute = data_change_show_df[data_change_show_df['Actual Action'] == 'Mute'].copy()
    tp_merged = data_change_show_df[(data_change_show_df['Dataset'] == 'TP') & 
                                    ~(data_change_show_df['Actual Action'].isin(['Create', 'Mute']))].copy()
    return ccx_create, ccx_update, ccx_expire, tp_create, tp_mute, tp_merged

def compute_changes_to_show(data_change_show_df, merged_df):
    """
    Compute changes to show in the UI based on the data change DataFrame and analyzed DataFrame.
    
    Args:
        data_change_show_df: DataFrame with changes to show from change simulation
        merged_df: DataFrame contains finalized item matching results to infor
    
    Returns:
        changes_simulation_result_df: DataFrame with changes to show in the UI
    """
    
    base_df = data_change_show_df.copy()

    im_df = merged_df[(merged_df['False Positive'] == False) & (merged_df['item_number_infor'] != '')].copy()
    im_df = im_df[['File_Row', 'item_number_infor', 'contract_number_infor']].copy()
    im_df.rename(columns = {'File_Row': 'File Row',
                            'item_number_infor': 'Item',
                            'contract_number_infor': 'Contract Number'}, inplace = True)
    im_df = im_df.drop_duplicates(subset=['File Row', 'Contract Number'])
    im_fr_mapping = dict(zip(im_df['File Row'], im_df['Item']))

    if im_df.empty:
        base_im_df = base_df.copy()
        base_im_df['Item'] = ''
        base_im_df['Item_fr'] = ''
    else:
        base_im_df = base_df.merge(im_df,
                                   on = ['File Row', 'Contract Number'],
                                   how = 'left')
    
        base_im_df['Item_fr'] = base_im_df['File Row'].map(im_fr_mapping)
        base_im_df['Item_fr'] = base_im_df['Item_fr'].apply(lambda x: '('+x+')' if not pd.isnull(x) else '')
        base_im_df['Item'] = base_im_df['Item'].fillna(base_im_df['Item_fr'])

    # take the portion of not no change to display
    changes_simulation_result_df = base_im_df[base_im_df['Actual Action'].isin(['Create', 
                                                                                'Update (New)',
                                                                                'Update (Existing)',
                                                                                'Expire then Create (Expire)',
                                                                                'Expire then Create (Create)',
                                                                                'Expire'])].copy()

    # add column to let user forgive the expiration of the item by mark 'Do Not Expire' as true (default to False)
    # for anything that are not set up as 'Expire CCX' under primary action, we will set it to nan
    changes_simulation_result_df['Do Not Expire'] = None
    changes_simulation_result_df.loc[changes_simulation_result_df['Primary Action'] == 'Expire CCX', 'Do Not Expire'] = False
    
    # extract the reference line for items to be expired
    file_rows_to_expire = set(changes_simulation_result_df[changes_simulation_result_df['Primary Action'] == 'Expire CCX']['File Row'])
    reference_for_expire_rows = base_im_df[(base_im_df['File Row'].isin(file_rows_to_expire)) & (base_im_df['Group'] == 'Keep')].copy()

    changes_simulation_result_df.to_excel(os.path.join(current_app.root_path, 'temp_files', 'changes_simulation_result_df.xlsx'), index=False)
    reference_for_expire_rows.to_excel(os.path.join(current_app.root_path, 'temp_files', 'reference_for_expire_rows.xlsx'), index=False)
    
    return changes_simulation_result_df, reference_for_expire_rows


def generate_network_graph(network_df, 
                           fixed_pos = None,
                           show_IUD = False):
    """
    Generate a network graph from the network DataFrame using Plotly
    Returns JSON string suitable for frontend consumption
    """
    if network_df.empty:
        return None
    
    # Prepare node sizes from total line counts
    df_a = network_df[['Contract Number_a', 'Total Contract Line Count_a']].rename(
        columns={'Contract Number_a': 'contract', 'Total Contract Line Count_a': 'total'}
    )
    df_b = network_df[['Contract Number_b', 'Total Contract Line Count_b']].rename(
        columns={'Contract Number_b': 'contract', 'Total Contract Line Count_b': 'total'}
    )
    
    # Filter out empty Contract Number_b values
    df_b = df_b[df_b['contract'].notna() & (df_b['contract'] != '')]
    
    df_nodes = pd.concat([df_a, df_b]).drop_duplicates('contract')
    
    # Clean and convert node sizes, handling NaN and invalid values
    df_nodes['total'] = pd.to_numeric(df_nodes['total'], errors='coerce').fillna(0)
    # CHANGE: Keep nodes with 0 total instead of filtering them out
    # df_nodes = df_nodes[df_nodes['total'] > 0]  # Remove this line
    
    node_sizes_map = dict(zip(df_nodes['contract'], df_nodes['total']))

    # change information mapping if show_IUD is True
    change_info_map = {}
    if show_IUD:
        # Process contract_a changes
        for _, row in network_df.iterrows():
            contract_a = row['Contract Number_a']
            if contract_a and contract_a != '':
                insert_a = row.get('Insert_Count_a', 0)
                update_a = row.get('Update_Count_a', 0)
                delete_a = row.get('Delete_Count_a', 0)
                
                if contract_a not in change_info_map:
                    change_info_map[contract_a] = {'Insert': 0, 'Update': 0, 'Delete': 0}
                
                change_info_map[contract_a]['Insert'] += insert_a
                change_info_map[contract_a]['Update'] += update_a
                change_info_map[contract_a]['Delete'] += delete_a
        
        # Process contract_b changes
        for _, row in network_df.iterrows():
            contract_b = row['Contract Number_b']
            if contract_b and contract_b != '':
                insert_b = row.get('Insert_Count_b', 0)
                update_b = row.get('Update_Count_b', 0)
                delete_b = row.get('Delete_Count_b', 0)
                
                if contract_b not in change_info_map:
                    change_info_map[contract_b] = {'Insert': 0, 'Update': 0, 'Delete': 0}
                
                change_info_map[contract_b]['Insert'] += insert_b
                change_info_map[contract_b]['Update'] += update_b
                change_info_map[contract_b]['Delete'] += delete_b

    # Track which contracts are contract_a for coloring
    a_contracts = set(network_df['Contract Number_a'])
    b_contracts = set(network_df['Contract Number_b'])
    
    # Build edge list (exclude self-links and empty Contract Number_b)
    edges = [
        (row['Contract Number_a'], row['Contract Number_b'], row['Overlapping Count'])
        for _, row in network_df.iterrows()
        if (pd.notna(row['Contract Number_b']) and 
            row['Contract Number_b'] != '' and
            row['Contract Number_a'] != row['Contract Number_b'] and 
            row['Overlapping Count'] > 0)
    ]

    # Create graph and explicitly add ALL nodes (including isolated ones)
    G = nx.Graph()
    
    # Add all contract nodes first (this ensures isolated nodes are included)
    for node in node_sizes_map.keys():
        if str(node).strip() != '' and str(node).strip().lower() != 'nan':
            G.add_node(node)
    
    # Then add edges (this won't affect isolated nodes)
    for u, v, w in edges:
        if u in node_sizes_map and v in node_sizes_map:  # Ensure both nodes exist
            G.add_edge(u, v, weight=w)
    
    # Check if graph is empty after filtering
    if len(G.nodes()) == 0:
        return None

    # Circular layout for nodes
    if fixed_pos is not None:
        # Use fixed positions if provided
        pos = {}
        for node in G.nodes():
            if node in fixed_pos:
                pos[node] = fixed_pos[node]
            else:
                # if new nodes not in fixed_pos, assign circular layout with no overlap
                temp_pos = nx.circular_layout([node])
                pos[node] = temp_pos[node]
    else:
        pos = nx.circular_layout(G)

    # Scale node sizes to a 10–40 range with better NaN handling
    sizes = list(node_sizes_map.values())
    # CHANGE: Only consider sizes > 0 for scaling, but handle 0 sizes separately
    valid_sizes = [s for s in sizes if not pd.isna(s) and s > 0]
    
    marker_sizes = {}
    if len(valid_sizes) > 1:
        min_size, max_size = np.log(min(valid_sizes)), np.log(max(valid_sizes))
        for n in G.nodes():
            node_total = node_sizes_map.get(n, 0)
            if pd.isna(node_total):
                node_total = 0
            
            # CHANGE: Handle 0 sizes specially
            if node_total == 0:
                marker_sizes[n] = 9  # Fixed small size for 0-count nodes
            else:
                if max_size > min_size:
                    scaled_size = 10 + (np.log(node_total) - min_size) / (max_size - min_size) * 30
                else:
                    scaled_size = 25  # Default size when all nodes are the same size
                
                marker_sizes[n] = max(10, min(40, scaled_size)) if not pd.isna(scaled_size) else 25
    else:
        # Handle case where all valid sizes are the same or no valid sizes
        for n in G.nodes():
            node_total = node_sizes_map.get(n, 0)
            if pd.isna(node_total) or node_total == 0:
                marker_sizes[n] = 15  # Fixed small size for 0-count nodes
            else:
                marker_sizes[n] = 25  # Default size

    # Edge traces
    edge_traces = []
    mid_x, mid_y, mid_text = [], [], []
    edge_hover_texts = []

    if len(network_df) > 0:
        # Clean overlapping counts
        overlap_counts = pd.to_numeric(network_df['Overlapping Count'], errors='coerce').fillna(0)
        overlap_counts = overlap_counts[overlap_counts >= 0]  # Remove negative values
        
        if len(overlap_counts) > 0:
            min_weight, max_weight = min(overlap_counts), max(overlap_counts)
            
            for u, v, d in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                w = d['weight']
                
                # Calculate line width safely
                if max_weight > min_weight and not pd.isna(w) and w >= 0:
                    line_width = 1 + (w - min_weight) / (max_weight - min_weight) * 20
                else:
                    line_width = 5
                
                edge_traces.append(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(width=max(1, min(21, line_width)), color='#888'),
                    hoverinfo='skip'
                ))
                
                mid_x.append((x0 + x1) / 2)
                mid_y.append((y0 + y1) / 2)
                mid_text.append(str(int(w)) if not pd.isna(w) else '0')
                edge_hover_texts.append(f"{u} ↔ {v}<br>Overlap: {int(w) if not pd.isna(w) else 0}")

    # Edge weight labels
    label_trace = go.Scatter(
        x=mid_x, y=mid_y,
        mode='markers+text',
        text=mid_text,
        textfont=dict(size=12, color='black', family='Arial'),
        texttemplate='%{text}',
        textposition='middle center',
        marker=dict(size=20, color='rgba(255, 255, 255, 0.8)', line=dict(width=0)),
        hoverinfo='text',
        hovertext=edge_hover_texts
    )

    # Self-loop arcs with better error handling
    self_map = {}
    try:
        for _, row in network_df[network_df['Contract Number_a'] == network_df['Contract Number_b']].iterrows():
            overlap = pd.to_numeric(row['Overlapping Count'], errors='coerce')
            if not pd.isna(overlap) and overlap > 0:
                self_map[row['Contract Number_a']] = int(overlap)
    except Exception as e:
        print(f"Error processing self-loops: {e}")

    # CHANGE: Color nodes with special handling for 0-count nodes
    node_colors = []
    for n in G.nodes():
        node_total = node_sizes_map.get(n, 0)
        
        # CHANGE: Red color for nodes with 0 total line count
        if node_total == 0:
            if n in b_contracts:
                node_colors.append('red')
            else:
                node_colors.append('grey')
        elif n in a_contracts:
            if n in self_map:
                # Self-looping contract_a nodes: translucent teal (mix of green and blue)
                node_colors.append('rgba(30, 200, 160, 0.9)')  # Teal with transparency
            else:
                # Non-self-looping contract_a nodes: translucent green
                node_colors.append('rgba(50, 210, 45, 0.9)')  # Green with transparency
        else:
            # Contract_b nodes: skyblue (no change)
            node_colors.append('skyblue')

    # Node trace with safe size values
    safe_marker_sizes = [marker_sizes.get(n, 25) for n in G.nodes()]
    # CHANGE: Handle 0 sizes in safe_node_sizes display
    safe_node_sizes = [int(node_sizes_map.get(n, 0)) for n in G.nodes()]

    # Enhanced hover text with change information if show_IUD is True
    node_hover_texts = []
    node_labels = []
    for i, n in enumerate(G.nodes()):
        base_text = f"{n}<br>Total lines: {safe_node_sizes[i]}"
        
        if show_IUD and n in change_info_map:
            changes = change_info_map[n]
            insert_count = changes.get('Insert', 0)
            update_count = changes.get('Update', 0)
            delete_count = changes.get('Delete', 0)

            change_text = f"<br>Changes: I:{insert_count} U:{update_count} D:{delete_count}"
            base_text += change_text

            node_labels.append(f"<span style='font-size:11px'>{n}</span><br><span style='font-size:10px'>I:{insert_count}, U:{update_count}, D:{delete_count}</span>")
        elif show_IUD:
            base_text += "<br>Changes: I:0 U:0 D:0"
            node_labels.append(f"<span style='font-size:11px'>{n}</span><br><span style='font-size:10px'>I:0, U:0, D:0</span>")
        else:
            node_labels.append(str(n))
        
        node_hover_texts.append(base_text)

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode='markers+text',
        text=node_labels,
        textposition='bottom center',
        marker=dict(
            size=safe_marker_sizes,
            color=node_colors,
            line=dict(width=2, color='#333')
        ),
        hoverinfo='text',
        hovertext=node_hover_texts,
        textfont=dict(size=11)
    )

    # Rest of the function remains the same...
    self_total_map = {}
    for n in self_map.keys():
        # Find the corresponding row in network_df for this contract
        self_loop_row = network_df[(network_df['Contract Number_a'] == n) & 
                                (network_df['Contract Number_b'] == n)]
        if not self_loop_row.empty:
            # Use Total Contract Line Count_b for the denominator
            total_b = pd.to_numeric(self_loop_row.iloc[0]['Total Contract Line Count_b'], errors='coerce')
            self_total_map[n] = max(1, int(total_b)) if not pd.isna(total_b) else 1
        else:
            # Fallback to Contract A total if B is not found
            total_a = node_sizes_map.get(n, 1)
            self_total_map[n] = max(1, int(total_a)) if not pd.isna(total_a) else 1

    arc_traces = []
    ring_mid_x, ring_mid_y, ring_mid_text = [], [], []
    ring_hover_texts = []

    for n, overlap in self_map.items():
        total = self_total_map[n]
        
        ratio = min(1.0, overlap / total) if total > 0 else 0
        arc_angle = max(0.2 * 2 * np.pi, ratio * 2 * np.pi)
        
        node_size = marker_sizes.get(n, 25)
        
        # Calculate radius based on node size with proper scaling
        base_radius = node_size / 200  # Convert pixel size to data coordinates
        min_radius = 0.12
        radius_buffer = base_radius * 0.15  # 15% larger than node
        
        # Ensure minimum radius and scale appropriately
        radius = max(min_radius, base_radius + radius_buffer)
        
        theta = np.linspace(0, arc_angle, 100)
        x_center, y_center = pos[n]
        arc_x = x_center + radius * np.cos(theta)
        arc_y = y_center + radius * np.sin(theta)
            
        arc_traces.append(go.Scatter(
            x=arc_x, y=arc_y,
            mode='lines',
            line=dict(width=2, color='rgba(46, 160, 44, 0.6)'),
            hoverinfo='skip'
        ))
        
        if ratio > 0.5:
            label_angle = arc_angle / 2
        else:
            label_angle = arc_angle * 0.9 if arc_angle > 0 else 0
            
        label_radius = radius * 1.3
        label_x = x_center + label_radius * np.cos(label_angle)
        label_y = y_center + label_radius * np.sin(label_angle)
        
        ring_mid_x.append(label_x)
        ring_mid_y.append(label_y)
        ring_mid_text.append(f"{overlap}/{total}")
        ring_hover_texts.append(f"{n} self-overlap: {overlap}/{total} ({ratio:.0%})")

    # Ring label background and text
    ring_background_trace = go.Scatter(
        x=ring_mid_x, y=ring_mid_y,
        mode='markers',
        marker=dict(size=25, color='rgba(255, 255, 255, 0.7)', line=dict(width=0)),
        hoverinfo='skip'
    )

    ring_label_trace = go.Scatter(
        x=ring_mid_x, y=ring_mid_y,
        mode='text',
        text=ring_mid_text,
        textfont=dict(size=10, color='#2ca02c'),
        hoverinfo='text',
        hovertext=ring_hover_texts
    )

    # Create figure
    fig_data = edge_traces + arc_traces + [label_trace, ring_background_trace, ring_label_trace, node_trace]
    
    layout = go.Layout(
        title='',
        hovermode='closest',
        margin=dict(l=15, r=15, t=30, b=80),
        annotations=[
            dict(
                text="Green nodes: Contract TP<br>Blue nodes: Contract CCX<br>Light green arcs/rings: overlaping contract(s) with overlap/total lines<br>Edges: Overlaps between contracts<br>Red/Grey nodes: Contracts with 0 lines",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.20,
                font=dict(size=10),
                bgcolor="rgba(255, 255, 255, 0.8)",
                borderpad=4,
                align="center"
            )
        ],
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            scaleanchor="y",
            scaleratio=1,
            range = [-1.3, 1.3]  # Adjust range to fit the circular layout
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range = [-1.3, 1.3]  # Adjust range to fit the circular layout
        ),
        showlegend=False,
        plot_bgcolor='rgba(245, 245, 252, 1)'
    )

    # Return JSON string
    return json.dumps({
        'data': fig_data,
        'layout': layout
    }, cls=PlotlyJSONEncoder)

class PlotlyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Plotly objects"""
    def default(self, obj):
        import numpy as np
        if hasattr(obj, 'to_plotly_json'):
            return obj.to_plotly_json()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)