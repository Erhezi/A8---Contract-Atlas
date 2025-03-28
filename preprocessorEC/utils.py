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
    Returns: (valid_df, error_df, has_errors)
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
    
    # Select only the required and error columns
    result_df = pd.DataFrame()
    for field in required_fields[:2] + ['Buyer Part Num'] + required_fields[2:]:
        if field in mapped_df.columns:
            result_df[field] = mapped_df[field]
        elif field != 'Buyer Part Num':  # Buyer Part Num is optional
            # Missing required column
            return None, None, f"Required field '{field}' is not mapped"
    
    # Create a copy for validation and error marking
    error_df = result_df.copy()
    
    # Add error columns
    error_df['Error-Missing Field'] = ''
    error_df['Error-Date'] = ''
    error_df['Error-Invalid Price'] = ''
    error_df['Error-Invalid QOE'] = ''
    error_df['Has Error'] = False
    
    # 1.2.1 Check for missing required fields
    for field in required_fields:
        if field == 'Buyer Part Num':
            continue  # Skip optional field
            
        # Mark rows where the field is empty
        error_df.loc[error_df[field].isna() | (error_df[field].str.strip() == ''), 'Error-Missing Field'] = 'Missing Data'
        error_df.loc[error_df[field].isna() | (error_df[field].str.strip() == ''), 'Has Error'] = True
    
    # 1.2.2 Validate date fields
    today_dt = datetime.now().date()
    today = today_dt.strftime('%Y-%m-%d')
    
    # Function to convert string to date safely
    # the input data usually follows 'YYYY-MM-DD' OR 'MM/DD/YYYY'
    def parse_date(date_str):
        if pd.isna(date_str) or date_str.strip() == '':
            return None
        
        # Try various date formats
        for fmt in ['%Y-%m-%d', '%m/%d/%Y']:
            try:
                # Return as string, not datetime object
                return datetime.strptime(date_str, fmt).date().strftime('%Y-%m-%d')
            except ValueError:
                pass
        return None
    
    # After parsing dates but before comparing them:
    def string_to_date(date_str):
        if pd.isna(date_str):
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        except:
            return None
        
    # Parse the date columns
    # Create new columns for parsed dates
    error_df['Effective_Date_Parsed'] = error_df['Effective Date'].apply(parse_date)
    error_df['Expiration_Date_Parsed'] = error_df['Expiration Date'].apply(parse_date)    
    
    # Convert string dates to actual date objects for comparison
    error_df['Effective_Date_Dt'] = error_df['Effective_Date_Parsed'].apply(string_to_date)
    error_df['Expiration_Date_Dt'] = error_df['Expiration_Date_Parsed'].apply(string_to_date)

    # Use these for comparisons
    valid_dates_mask = (~error_df['Effective_Date_Dt'].isna()) & (~error_df['Expiration_Date_Dt'].isna())

    # Compare date objects, not strings
    error_df.loc[valid_dates_mask & (error_df['Effective_Date_Dt'] < today_dt), 'Error-Date'] = 'Effective Date must be >= today'
    error_df.loc[valid_dates_mask & (error_df['Effective_Date_Dt'] < today_dt), 'Has Error'] = True

    
    # Expiration date should be >= today and >= effective date
    # For expiration date
    invalid_exp_mask = valid_dates_mask & (
        (error_df['Expiration_Date_Dt'] < today_dt) | 
        (error_df['Expiration_Date_Dt'] < error_df['Effective_Date_Dt'])
    )
    error_df.loc[invalid_exp_mask, 'Error-Date'] = 'Expiration Date must be >= Effective Date and today'
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
    
    # Drop temporary columns used for validation
    # Don't forget to drop these new columns too
    error_df.drop(['Effective_Date_Parsed', 'Expiration_Date_Parsed', 
               'Effective_Date_Dt', 'Expiration_Date_Dt', 
               'Price_Parsed', 'QOE_Parsed'], axis=1, inplace=True)
    
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
