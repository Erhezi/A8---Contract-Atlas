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

def validate_file(df, column_mapping):
    """
    Validate the file contents against the required fields
    Returns: (result_df, error_df, has_errors)
    """
    # Required fields (all required except 'Buyer Part Num')
    required_fields = [
        'Mfg Part Num', 'Vendor Part Num', 'Description', 
        'Contract Price', 'UOM', 'QOE', 'Effective Date', 'Expiration Date', 
        'Contract Number', 'ERP Vendor ID'
    ]
    
    # Create a copy of the dataframe with standard field names
    mapped_df = df.copy()
    
    # Rename columns according to the mapping
    for std_field, user_field in column_mapping.items():
        if user_field in mapped_df.columns:
            mapped_df.rename(columns={user_field: std_field}, inplace=True)
        if 'Buyer Part Num' not in mapped_df.columns:
            mapped_df.loc[:, 'Buyer Part Num'] = ''

    # making a reduced form of Mfg Part Num, so leading zeros and dash are removed from the Mfg Part Num
    def reduce_mfg_part_num(mfg_part_num):
        if pd.isnull(mfg_part_num) or mfg_part_num.strip() == '':
            return np.nan
        mfg_part_num = mfg_part_num.strip().replace('-', '').strip()
        if mfg_part_num.isdigit():
            return str(int(mfg_part_num))
        return mfg_part_num
    
    # adding two more columns for later use 'Original UOM' and 'Reduced Mfg Part Num'
    mapped_df['Original UOM'] = mapped_df['UOM'].apply(lambda x: x.strip().upper())
    mapped_df['Reduced Mfg Part Num'] = mapped_df['Mfg Part Num'].apply(lambda x: reduce_mfg_part_num(x))
    
    # Select only the required and error columns
    missing_fields = []
    for field in required_fields:
        if field not in mapped_df.columns:
            missing_fields.append(field)
    if missing_fields != []:
        return None, None, f"Required field(s) {', '.join(missing_fields)} not found"
    
    # Create a copy for validation and error marking
    result_df = mapped_df[required_fields[:2] + ['Buyer Part Num'] + required_fields[2:] + ['Reduced Mfg Part Num', 'Original UOM']].copy()
    error_df = result_df.copy()
    
    # Add error columns
    error_df['Error-Missing Field'] = ''
    error_df['Error-Invalid Date'] = ''
    error_df['Error-Invalid Price'] = ''
    error_df['Error-Invalid QOE'] = ''
    error_df['Error-Invalid UOM'] = ''
    error_df['Error-EA QOE NOT 1'] = ''
    error_df['Has Error'] = False
    
    # 1.2.1 Validate all requried fields has value
    for field in required_fields:
        missing_mask = error_df[field].isna() | (error_df[field].str.strip() == '')
        error_df.loc[missing_mask, 'Error-Missing Field'] = 'Missing Data'
        error_df.loc[missing_mask, 'Has Error'] = True

    # 1.2.2 Validate date fields
    today_dt = datetime.now().date()

    # Convert date strings to datetime objects, invalid dates become NaT
    error_df['Effective_Date_Dt'] = pd.to_datetime(error_df['Effective Date'], errors='coerce')
    error_df['Expiration_Date_Dt'] = pd.to_datetime(error_df['Expiration Date'], errors='coerce')

    # Create YYYY-MM-DD string versions for the output
    error_df['Effective Date'] = error_df['Effective_Date_Dt'].dt.strftime('%Y-%m-%d')
    error_df['Expiration Date'] = error_df['Expiration_Date_Dt'].dt.strftime('%Y-%m-%d')

    # Mark rows with unparseable dates
    error_df.loc[error_df['Effective_Date_Dt'].isna() & ~(error_df['Effective Date'].isna() | (error_df['Effective Date'].str.strip() == '')), 'Error-Date'] = 'Invalid Effective Date format'
    error_df.loc[error_df['Effective_Date_Dt'].isna() & ~(error_df['Effective Date'].isna() | (error_df['Effective Date'].str.strip() == '')), 'Has Error'] = True

    error_df.loc[error_df['Expiration_Date_Dt'].isna() & ~(error_df['Expiration Date'].isna() | (error_df['Expiration Date'].str.strip() == '')), 'Error-Date'] = 'Invalid Expiration Date format'
    error_df.loc[error_df['Expiration_Date_Dt'].isna() & ~(error_df['Expiration Date'].isna() | (error_df['Expiration Date'].str.strip() == '')), 'Has Error'] = True

    # Skip validation for already marked error rows
    valid_dates_mask = (~error_df['Effective_Date_Dt'].isna()) & (~error_df['Expiration_Date_Dt'].isna())

    # # Validate effective date >= today
    # invalid_eff_mask = valid_dates_mask & (error_df['Effective_Date_Dt'].dt.date < today_dt)
    # error_df.loc[invalid_eff_mask, 'Error-Invalid Date'] = 'Effective Date must be >= today'
    # error_df.loc[invalid_eff_mask, 'Has Error'] = True

    # Validate expiration date >= today and >= effective date
    invalid_exp_mask = valid_dates_mask & (
        (error_df['Expiration_Date_Dt'].dt.date < today_dt) | 
        (error_df['Expiration_Date_Dt'] < error_df['Effective_Date_Dt'])
    )
    error_df.loc[invalid_exp_mask, 'Error-Invalid Date'] = 'Expiration Date must be >= Effective Date and today'
    error_df.loc[invalid_exp_mask, 'Has Error'] = True
    
    # 1.2.3 Validate Contract Price
    def convert_price(price_str):
        if pd.isna(price_str) or price_str.strip() == '':
            return None
        try:
            # Remove $ and commas
            cleaned = re.sub(r'[,$]', '', price_str.strip())
            return float(cleaned)
        except:
            return None
    
    error_df['Price_Parsed'] = error_df['Contract Price'].apply(convert_price)
    error_df.loc[error_df['Price_Parsed'].isna(), 'Error-Invalid Price'] = 'Price not Recognized'
    error_df.loc[error_df['Price_Parsed'].isna(), 'Has Error'] = True
    
    # 1.2.4 Validate QOE (Quantity of Each)
    def convert_qoe(qoe_str):
        if pd.isna(qoe_str) or qoe_str.strip() == '':
            return None
        try:
            return int(qoe_str.strip())
        except:
            return None
    
    error_df['QOE_Parsed'] = error_df['QOE'].apply(convert_qoe)
    error_df.loc[error_df['QOE_Parsed'].isna(), 'Error-Invalid QOE'] = 'QOE not Recognized'
    error_df.loc[error_df['QOE_Parsed'].isna(), 'Has Error'] = True
    
    # 1.2.5 Validate UOM (Units of Measure)
    # First, save the original UOM values before converting to uppercase
    error_df['Original UOM'] = error_df['UOM'].apply(lambda x: np.nan if pd.isnull(x) else x.strip().upper())
    
    # Load the UOM mapping file
    try:
        uom_file_path = os.path.join(current_app.root_path, 'preprocessorEC', 'data', 'UOM.csv')
        uom_df = pd.read_csv(uom_file_path)
        
        # Create UOM dictionary mapping from "see UOM" to "use UOM"
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
        # If there's an error with the UOM file, we don't fail the whole validation
        # Just log the error and continue with other validations
    
    # 1.2.6 Validate UOM-QOE Compatibility
    # when UOM == 'EA', QOE should be 1
    error_df.loc[(error_df['UOM'] == 'EA') & (error_df['QOE_Parsed'] != 1), 'Error-EA QOE NOT 1'] = 'QOE must be 1 for UOM EA'
    error_df.loc[(error_df['UOM'] == 'EA') & (error_df['QOE_Parsed'] != 1), 'Has Error'] = True
    
    # Drop temporary columns used for validation
    temp_columns = ['Effective_Date_Dt', 'Expiration_Date_Dt', 
                   'Price_Parsed', 'QOE_Parsed']
    
    # Only drop columns that exist in the dataframe
    columns_to_drop = [col for col in temp_columns if col in error_df.columns]
    if columns_to_drop:
        error_df.drop(columns_to_drop, axis=1, inplace=True)
    
    # Check if there are any errors
    has_errors = error_df['Has Error'].any()
    
    # If there are errors, return the error dataframe
    # Otherwise, return the cleaned dataframe for further processing
    if has_errors:
        return result_df, error_df, has_errors
    else:
        # If no errors, return the valid dataframe without error columns
        return result_df, None, has_errors

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
