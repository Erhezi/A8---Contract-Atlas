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

# Global model cache
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
        'Contract Number', 'ERP Vendor ID', 'Source Contract Type'
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
    error_df['Warning-Potential Duplicates'] = ''
    error_df['Has Error'] = False
    
    return error_df, columns_to_save_to_session, missing_fields, required_fields

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
    
    return error_df

def validate_required_fields(error_df, required_fields):
    """Check that all required fields have values"""
    for field in required_fields:
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

    # Validate expiration date >= today and > effective date
    invalid_exp_mask = ~empty_date_mask & (
        (error_df['Expiration_Date_Dt'].dt.date < today_dt) | 
        (error_df['Expiration_Date_Dt'] <= error_df['Effective_Date_Dt'])
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
        # Use distributor keys: ERP Vendor ID + Mfg Part Num + UOM
        duplicate_keys = ['ERP Vendor ID', 'Mfg Part Num', 'UOM']
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
    error_df = validate_required_fields(error_df, required_fields)
    error_df = validate_source_contract_type(error_df)  # Add new validation step
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
            'Dataset': 'CCX',
            'File Row': item.get('File_Row', ''),  # Group identifier
            'Pair ID': 'tc' + str(i) # Unique identifier for the pair
        }
        ccx_data.append(ccx_row)
    
    # Create upload dataframe
        upload_row = {
            'Source Contract Type': item.get('Source_Contract_Type', ''),
            'Contract Number': item.get('Contract_Number', ''),
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
    if merged_df.empty:
        return pd.DataFrame()
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
            'Dataset': 'Infor',
            'File Row': item.get('File_Row', ''),
            'Pair ID': 'ti' + str(p_cnt), # Unique identifier for the pair
            'ItemNumber': item.get('item_number_infor', ''),
            'Contract ERP ID': item.get('erp_contract_id_infor', '') # same contract number sometimes can be put under two different vendor, thus two contract objects
        }
        tp_row = {
            'Source Contract Type': item.get('Source_Contract_Type', ''),
            'Contract Number': item.get('Contract_Number', ''),
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
        None
    """
    # merge dataframe to have the indicator information aligned
    df_m = validated_df.merge(stacked_df,
                              on = ['File Row'],
                              how = 'left',
                              suffixes = ('_a', '_b'),
                              indicator = True)
    
    # fill na on b side to soak a side informations
    for col in ['Source Contract Type', 'Contract Number', 'Reduced Mfg Part Num', 'Mfg Part Num', 
                'Vendor Part Num', 'Buyer Part Num', 'Description', 'UOM', 'QOE', 
                'Contract Price', 'Effective Date', 'Expiration Date']:
        if col + '_b' in df_m.columns:
            df_m[col + '_b'] = df_m[col + '_b'].fillna(df_m[col + '_a'])

    # calculate EA price
    df_m['EA Price'] = df_m['Contract Price_b']/df_m['QOE_b']
    # fill for other columns
    df_m['Dataset'] = df_m['Dataset'].fillna('TP')
    df_m['Rank'] = df_m['Rank'].fillna(1)
    df_m['Keep'] = df_m['Keep'].fillna(True)
    
    # mark for same contract number (make sure they are upper cased and stripped)
    df_m['Contract Number_a'] = df_m['Contract Number_a'].astype(str).str.strip().str.upper()
    df_m['Contract Number_b'] = df_m['Contract Number_b'].astype(str).str.strip().str.upper()
    df_m['Same Contract Number'] = df_m['Contract Number_a']== df_m['Contract Number_b']
    
    return df_m
    