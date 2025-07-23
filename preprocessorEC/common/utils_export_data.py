import pandas as pd
from datetime import datetime

def make_batch_upload_excel(batch_df, batch_filepath, skip_gpo):
    """
    Prepares the batch upload excel file by ensuring it has the correct columns and data types.
    
    Args:
        batch_df (pd.DataFrame): The DataFrame to prepare for batch upload.
        batch_filepath: the path that temporarily stores the output file before zipping with other files
        
    Returns:
        bool: True if the DataFrame is successfully prepared, False otherwise.
        error_msg (str): Error message if preparation fails, empty string otherwise.
    """
    # config for columns
    columns_no_highlight = [
        'Organization', 'Vendor', 'Manufacturer', 'Contract Number',
        'Contract Description', 'Tier Level', 'Tier Description', 'Source Type',
        'Start Date', 'End Date', 'Mfg Part Num', 'Vendor Part Num',
        'Buyer Part Num', 'Description', 'Contract Price', 'UOM', 'QOE',
        'Effective Date', 'Expiration Date'
    ]
    columns_with_highlight = [
        'Actual Action',  # Added Actual Action for reference
        'Execute Order'   # Add to help find indexing later in coloring
    ]
    all_columns = columns_no_highlight + columns_with_highlight
    final_renaming = {'Mfg Part Num': 'Mfg Part',
                      'Vendor Part Num': 'Vendor Part',
                      'Buyer Part Num': 'Buyer Part',
                      'Contract Price': 'Price',
                      'QOE': 'QTY'}
    try:
        if batch_df.empty:
            # batch_df can be empty, in this case we just write an empty file and return True
            batch_df = pd.DataFrame(columns=columns_no_highlight)
            batch_df.rename(columns = final_renaming, inplace=True)
            # write this to the batch_filepath and return true
            batch_df.to_excel(batch_filepath, index=False)
            return True, ""
        
        # Split GPO records if needed
        if 'Source Type' in batch_df.columns:
            if skip_gpo:
                batch_df = batch_df[~batch_df['Source Type'].str.startswith('GPO', na=False)].copy()
        
        # Sort by Actual Action as requested
        if 'Actual Action' in batch_df.columns:
            action_order = {
                'Expire': 0,
                'Expire then Create (Expire)': 1,
                'Expire then Create (Create)': 2,
                'Update (New)': 3,
                'Create': 4
            }
            
            # Create a sorting key based on the action order
            batch_df['action_sort'] = batch_df['Actual Action'].apply(
                lambda x: action_order.get(x, 999)  # Default high number for unknown actions
            )
            batch_df = batch_df.sort_values('action_sort').drop('action_sort', axis=1)
        
        # Preprocess batch_df for date formatting and sorting
        date_columns = ['Start Date', 'End Date', 'Effective Date', 'Expiration Date']

        # Format date columns in batch_df
        for col in date_columns:
            if col in batch_df.columns:
                batch_df[col] = pd.to_datetime(batch_df[col], errors='coerce')

        # Add helper column for row coloring
        batch_df['Execute Order'] = range(1, len(batch_df) + 1)
        
        # Select columns that exist in the DataFrame
        available_columns = [col for col in all_columns if col in batch_df.columns]
        batch_output = batch_df[available_columns].copy()
        
        # Find rows with date conflicts, missing vendor parts, or duplicate creations
        date_conflict_rows = []
        missing_vendor_rows = []
        double_create_rows = []
        
        for i, row in batch_df.iterrows():
            # Check for date conflicts
            if (('Final L Date Check' in row and row['Final L Date Check'] != 'pass') or 
                ('Final H Date Check' in row and row['Final H Date Check'] != 'pass')):
                date_conflict_rows.append(row['Execute Order'])
            
            # Check for missing vendor parts with specific action
            if ((pd.isna(row.get('Vendor Part Num')) or row.get('Vendor Part Num') == '') and 
                row.get('Actual Action') in ['Create', 'Expire then Create (Create)', 'Update (New)']):
                missing_vendor_rows.append(row['Execute Order'])
                
            # Check for possible duplicates (Create Count > 1)
            if 'Create Count' in row and pd.notna(row['Create Count']) and int(row['Create Count']) > 1:
                double_create_rows.append(row['Execute Order'])

        # rename some columns before exporting
        batch_output = batch_output.rename(columns=final_renaming)

        # if date_conflict_rows or missing_vendor_rows or double_create_rows not empty, apply formatting
        if date_conflict_rows or double_create_rows or missing_vendor_rows:
            with pd.ExcelWriter(batch_filepath, engine='xlsxwriter', datetime_format='m/d/yyyy') as writer:
                batch_output.to_excel(writer, sheet_name='Batch Upload', index=False)

                workbook = writer.book
                worksheet = writer.sheets['Batch Upload']

                # Define formats
                red_format = workbook.add_format({'bg_color': '#f8d7da'})
                purple_format = workbook.add_format({'bg_color': '#e2d9f3'})
                yellow_format = workbook.add_format({'bg_color': '#fff3cd'})
                red_date_format = workbook.add_format({'bg_color': '#f8d7da', 'num_format': 'm/d/yyyy'})
                purple_date_format = workbook.add_format({'bg_color': '#e2d9f3', 'num_format': 'm/d/yyyy'})
                yellow_date_format = workbook.add_format({'bg_color': '#fff3cd', 'num_format': 'm/d/yyyy'})

                # Map column indices
                col_indices = {col: idx for idx, col in enumerate(batch_output.columns)}
                date_cols = [col for col in date_columns if col in col_indices]

                # Apply formatting row-by-row
                for row_idx, row in batch_output.iterrows():
                    excel_row = row['Execute Order']  # Excel row (1-based header offset)
                    row_number = excel_row  # already adjusted

                    if excel_row in date_conflict_rows:
                        base_format = red_format
                        date_format = red_date_format
                    elif excel_row in double_create_rows:
                        base_format = purple_format
                        date_format = purple_date_format
                    elif excel_row in missing_vendor_rows:
                        base_format = yellow_format
                        date_format = yellow_date_format
                    else:
                        continue  # skip non-highlighted rows

                    for col_idx, col_name in enumerate(batch_output.columns):
                        value = row[col_name]

                        if col_name in date_cols and pd.notna(value):
                            worksheet.write_datetime(row_number, col_idx, pd.to_datetime(value), date_format)
                        else:
                            worksheet.write(row_number, col_idx, value, base_format)
        
        else:
            # if no issue, drop the helper column (column with highlight) and write to file
            batch_output = batch_output.drop(columns=columns_with_highlight, errors='ignore')
            with pd.ExcelWriter(batch_filepath, engine='xlsxwriter', datetime_format='m/d/yyyy') as writer:
                batch_output.to_excel(writer, sheet_name='Batch Upload', index=False)
    
    except Exception as e:
        error_msg = f"Error preparing batch upload DataFrame: {str(e)}"
        print(error_msg)
        return False, error_msg
    
    return True, ""


def make_batch_upload_csv(batch_df, batch_filepath_csv, skip_gpo):
    """
    Prepares the batch upload csv by ensuring it has the correct columns and data types.
    
    Args:
        batch_df (pd.DataFrame): The DataFrame to prepare for batch upload.
        batch_filepath: the path that temporarily stores the output file before zipping with other files
        
    Returns:
        bool: True if the DataFrame is successfully prepared, False otherwise.
        error_msg (str): Error message if preparation fails, empty string otherwise.
    """
    try:
        if batch_df.empty:
            # batch_df can be empty, in this case we just write an empty file and return True
            batch_df = pd.DataFrame(columns=['Organization', 'Vendor', 'Manufacturer', 'Contract Number',
                                             'Contract Description', 'Tier Level', 'Tier Description', 'Source Type',
                                             'Start Date', 'End Date', 'Mfg Part Num', 'Vendor Part Num',
                                             'Buyer Part Num', 'Description', 'Contract Price', 'UOM', 'QOE',
                                             'Effective Date', 'Expiration Date'])
            # write this to the batch_filepath_csv and return true
            batch_df.to_csv(batch_filepath_csv, index=False)
            return True, ""
        
        # Split GPO records if needed
        if 'Source Type' in batch_df.columns:
            if skip_gpo:
                batch_df = batch_df[~batch_df['Source Type'].str.startswith('GPO', na=False)].copy()
        
        # Sort by Actual Action as requested
        if 'Actual Action' in batch_df.columns:
            action_order = {
                'Expire': 0,
                'Expire then Create (Expire)': 1,
                'Expire then Create (Create)': 2,
                'Update (New)': 3,
                'Create': 4
            }
            
            # Create a sorting key based on the action order
            batch_df['action_sort'] = batch_df['Actual Action'].apply(
                lambda x: action_order.get(x, 999)  # Default high number for unknown actions
            )
            batch_df = batch_df.sort_values('action_sort').drop('action_sort', axis=1)
        
        # add execute order column
        batch_df['Execute Order'] = range(1, len(batch_df) + 1)
        
        # csv will output all needed columns including the ones in excel with highlight and additional date check results
        all_columns = [
            'Organization', 'Vendor', 'Manufacturer', 'Contract Number',
            'Contract Description', 'Tier Level', 'Tier Description', 'Source Type',
            'Start Date', 'End Date', 'Mfg Part Num', 'Vendor Part Num',
            'Buyer Part Num', 'Description', 'Contract Price', 'UOM', 'QOE',
            'Effective Date', 'Expiration Date', 
            'Actual Action', 
            'Execute Order',
            'Final L Date Check', 'Final H Date Check', 'Create Count', 'ERP Vendor ID (CCX Sync)'
        ]

        batch_df = batch_df[all_columns].copy()

        # Format date columns in batch_df
        date_columns = ['Start Date', 'End Date', 'Effective Date', 'Expiration Date']
        for col in date_columns:
            if col in batch_df.columns:
                batch_df[col] = pd.to_datetime(batch_df[col], errors='coerce')

        # export as csv
        batch_df.to_csv(batch_filepath_csv, index=False, date_format='%m/%d/%Y')
        
    except Exception as e:
        error_msg = f"Error preparing batch upload CSV: {str(e)}"
        print(error_msg)
        return False, error_msg
    
    return True, ""


def make_infor_direct_excel(gpo_df, infor_direct_filepath):
    """
    Prepares the Infor Direct upload excel file by ensuring it has the correct columns and data types.
    the output excel will typically have two sheets: 
    - GPO data: one for infor direct that contains changes for all GPO records
    - GHX_EXCLUD_GLD_add: another only takes ['Contract Number', 'Mfg Part Num', 'Supplier (CCX Sync)', 'UOM']
      with acutal action 'Update (New)', Expire then Create (Expire)' and 'Expire'
    
    Args:
        gpo_df (pd.DataFrame): The DataFrame to prepare for Infor Direct upload.
        infor_direct_filepath: the path that temporarily stores the output file before zipping with other files
        
    Returns:
        bool: True if the DataFrame is successfully prepared, False otherwise.
        error_msg (str): Error message if preparation fails, empty string otherwise.
    """
    data_tab_columns = ['Organization', 'Vendor', 'Manufacturer', 'Contract Number',
                        'Contract Description', 'Tier Level', 'Tier Description', 'Source Type',
                        'Start Date', 'End Date', 'Mfg Part Num', 'Vendor Part Num',
                        'Buyer Part Num', 'Description', 'Contract Price', 'UOM', 'QOE',
                        'Effective Date', 'Expiration Date',
                        'Actual Action', 'ERP Vendor ID (CCX Sync)', 'Supplier (CCX Sync)',
                        'ERP Vendor ID (PrP)', 'Supplier (PrP)', 'Item', 'Link Item Flag',
                        'TaskID', 'UserID', 'Final L Date Check', 'Final H Date Check', 'Create Count']
    ghx_exclud_gld_add_columns = ['Contract Number', 'Mfg Part Num', 'Supplier (CCX Sync)', 'UOM']
    
    try:
        if gpo_df.empty:
            # batch_df can be empty, in this case we just write an empty file and return True
            gpo_df = pd.DataFrame(columns=data_tab_columns)
            # write this to the infor_direct_filepath and return true
            gpo_df.to_excel(infor_direct_filepath, index=False)
            return True, ""
        
        # formate date columns
        date_columns = ['Start Date', 'End Date', 'Effective Date', 'Expiration Date']
        for col in date_columns:
            if col in gpo_df.columns:
                gpo_df[col] = pd.to_datetime(gpo_df[col], errors='coerce')
        
        # sort
        if 'Actual Action' in gpo_df.columns:
            action_order = {
                'Expire': 0,
                'Expire then Create (Expire)': 1,
                'Expire then Create (Create)': 2,
                'Update (New)': 3,
                'Create': 4
            }
            
            # Create a sorting key based on the action order
            gpo_df['action_sort'] = gpo_df['Actual Action'].apply(
                lambda x: action_order.get(x, 999)  # Default high number for unknown actions
            )
            gpo_df = gpo_df.sort_values('action_sort').drop('action_sort', axis=1)
        

        # Ensure all required columns are present
        data_columns = [col for col in data_tab_columns if col in gpo_df.columns]
        data_output = gpo_df[data_columns].copy()
        gld_columns = [col for col in ghx_exclud_gld_add_columns if col in gpo_df.columns]
        gld_output = gpo_df[gld_columns].copy()
       
        # Write to Excel file
        with pd.ExcelWriter(infor_direct_filepath, engine='xlsxwriter', datetime_format='m/d/yyyy') as writer:
            data_output.to_excel(writer, sheet_name='Data', index=False)
            gld_output.to_excel(writer, sheet_name='GHX_EXCLUD_GLD_add', index=False)

    except Exception as e:
        error_msg = f"Error preparing Infor Direct upload DataFrame: {str(e)}"
        print(error_msg)
        return False, error_msg
    
    return True, ""


def make_infor_direct_csv1(gpo_df, infor_direct_filepath_csv1):
    """Prepares the Infor Direct upload csv by ensuring it has the correct columns and data types.
    Args:
        gpo_df (pd.DataFrame): The DataFrame to prepare for Infor Direct upload.
        infor_direct_filepath_csv: the path that temporarily stores the output file before zipping with other files
    Returns:
        bool: True if the DataFrame is successfully prepared, False otherwise.
        error_msg (str): Error message if preparation fails, empty string otherwise.
    """
    data_tab_columns = ['Organization', 'Vendor', 'Manufacturer', 'Contract Number',
                        'Contract Description', 'Tier Level', 'Tier Description', 'Source Type',
                        'Start Date', 'End Date', 'Mfg Part Num', 'Vendor Part Num',
                        'Buyer Part Num', 'Description', 'Contract Price', 'UOM', 'QOE',
                        'Effective Date', 'Expiration Date',
                        'Actual Action', 'ERP Vendor ID (CCX Sync)', 'Supplier (CCX Sync)',
                        'ERP Vendor ID (PrP)', 'Supplier (PrP)', 'Item', 'Link Item Flag',
                        'TaskID', 'UserID', 'Final L Date Check', 'Final H Date Check', 'Create Count']
    
    try:
        if gpo_df.empty:
            # batch_df can be empty, in this case we just write an empty file and return True
            gpo_df = pd.DataFrame(columns=data_tab_columns)
            # write this to the infor_direct_filepath_csv and return true
            gpo_df.to_csv(infor_direct_filepath_csv1, index=False)
            return True, ""
        
        # formate date columns
        date_columns = ['Start Date', 'End Date', 'Effective Date', 'Expiration Date']
        for col in date_columns:
            if col in gpo_df.columns:
                gpo_df[col] = pd.to_datetime(gpo_df[col], errors='coerce')

        # Ensure all required columns are present
        data_columns = [col for col in data_tab_columns if col in gpo_df.columns]
        data_output = gpo_df[data_columns].copy()
        
        # export as csv
        data_output.to_csv(infor_direct_filepath_csv1, index=False, date_format='%m/%d/%Y')
        
    except Exception as e:
        error_msg = f"Error preparing Infor Direct upload CSV: {str(e)}"
        print(error_msg)
        return False, error_msg
    
    return True, ""


def make_infor_direct_csv2(gpo_df, infor_direct_filepath_csv2):
    """Prepares the Infor Direct upload csv for GHX_EXCLUD_GLD_add by ensuring it has the correct columns and data types.
    
    Args:
        gpo_df (pd.DataFrame): The DataFrame to prepare for Infor Direct upload.
        infor_direct_filepath_csv2: the path that temporarily stores the output file before zipping with other files
        
    Returns:
        bool: True if the DataFrame is successfully prepared, False otherwise.
        error_msg (str): Error message if preparation fails, empty string otherwise.
    """
    ghx_exclud_gld_add_columns = ['Contract Number', 'Mfg Part Num', 'Supplier (CCX Sync)', 'UOM']
    
    try:
        if gpo_df.empty:
            # batch_df can be empty, in this case we just write an empty file and return True
            gpo_df = pd.DataFrame(columns=ghx_exclud_gld_add_columns)
            # write this to the infor_direct_filepath_csv2 and return true
            gpo_df.to_csv(infor_direct_filepath_csv2, index=False)
            return True, ""
        
        # Ensure all required columns are present
        gld_columns = [col for col in ghx_exclud_gld_add_columns if col in gpo_df.columns]
        gld_output = gpo_df[gld_columns].copy()
        
        # export as csv
        gld_output.to_csv(infor_direct_filepath_csv2, index=False)
        
    except Exception as e:
        error_msg = f"Error preparing Infor Direct upload CSV for GHX_EXCLUD_GLD_add: {str(e)}"
        print(error_msg)
        return False, error_msg
    
    return True, ""


def make_single_contract_excel(single_df, single_filepath, contract_number):
    """
    Prepares a single contract excel file by ensuring it has the correct columns and data types.
    the single_df can contain multiple contract, we will filter it by contract_number
    
    Args:
        single_df (pd.DataFrame): The DataFrame to prepare for single contract upload.
        single_filepath: the path that temporarily stores the output file before zipping with other files
        
    Returns:
        bool: True if the DataFrame is successfully prepared, False otherwise.
        error_msg (str): Error message if preparation fails, empty string otherwise.
    """
    # config for columns
    columns_no_highlight = [
        'Mfg Part Num', 'Vendor Part Num',
        'Buyer Part Num', 'Description', 'Contract Price', 'UOM', 'QOE',
        'Effective Date', 'Expiration Date'
    ]
    columns_with_highlight = [
        'Actual Action',  # Added Actual Action for reference
        'ERP vendor ID (PrP)', # Added ERP vendor ID for reference
    ]
    all_columns = columns_no_highlight + columns_with_highlight
    try:
        if single_df.empty:
            # single_df can be empty, in this case we just write an empty file and return True
            single_df = pd.DataFrame(columns=columns_no_highlight)
            # write this to the single_filepath and return true
            single_df.to_excel(single_filepath, index=False)
            return True, ""
        
        # Filter by contract number
        contract_df = single_df[single_df['Contract Number (PrP)'] == contract_number].copy().reset_index(drop=True)
        if contract_df.empty:
            error_msg = f"No records found for contract number: {contract_number}"
            print(error_msg)
            return False, error_msg
        
        # format date columns
        date_columns = ['Effective Date', 'Expiration Date']
        for col in date_columns:
            if col in contract_df.columns:
                contract_df[col] = pd.to_datetime(contract_df[col], errors='coerce')
        

        # Select columns that exist in the DataFrame
        available_columns = [col for col in all_columns if col in contract_df.columns]
        contract_output = contract_df[available_columns].copy()

        # find rows with date conflicts, missing vendor parts, or duplicate creations
        date_conflict_rows = []
        missing_vendor_rows = []
        double_create_rows = []
        for i, row in contract_df.iterrows():
            # Check for date conflicts
            if (('Final L Date Check' in row and row['Final L Date Check'] != 'pass') or 
                ('Final H Date Check' in row and row['Final H Date Check'] != 'pass')):
                date_conflict_rows.append(i+1)
            
            # Check for missing vendor parts with specific action
            if ((pd.isna(row.get('Vendor Part Num')) or row.get('Vendor Part Num') == '') and 
                row.get('Actual Action') in ['Create', 'Expire then Create (Create)', 'Update (New)']):
                missing_vendor_rows.append(i+1)
                
            # Check for possible duplicates (Create Count > 1)
            if 'Create Count' in row and int(row['Create Count']) > 1:
                double_create_rows.append(i+1)
        

        # if we have highlight to apply
        if date_conflict_rows or double_create_rows or missing_vendor_rows:
            with pd.ExcelWriter(single_filepath, engine='xlsxwriter', datetime_format='m/d/yyyy') as writer:
                contract_output.to_excel(writer, sheet_name='Single Contract Upload', index=False)

                workbook = writer.book
                worksheet = writer.sheets['Single Contract Upload']

                # Define formats
                red_format = workbook.add_format({'bg_color': '#f8d7da'})
                purple_format = workbook.add_format({'bg_color': '#e2d9f3'})
                yellow_format = workbook.add_format({'bg_color': '#fff3cd'})
                red_date_format = workbook.add_format({'bg_color': '#f8d7da', 'num_format': 'm/d/yyyy'})
                purple_date_format = workbook.add_format({'bg_color': '#e2d9f3', 'num_format': 'm/d/yyyy'})
                yellow_date_format = workbook.add_format({'bg_color': '#fff3cd', 'num_format': 'm/d/yyyy'})

                # Map column indices
                col_indices = {col: idx for idx, col in enumerate(contract_output.columns)}
                date_cols = [col for col in date_columns if col in col_indices]

                # Apply formatting row-by-row
                for row_idx, row in contract_output.iterrows():
                    excel_row = row_idx + 1 # Excel row (1-based header offset)
                    row_number = row_idx + 1  # already adjusted

                    if excel_row in date_conflict_rows:
                        base_format = red_format
                        date_format = red_date_format
                    elif excel_row in double_create_rows:
                        base_format = purple_format
                        date_format = purple_date_format
                    elif excel_row in missing_vendor_rows:
                        base_format = yellow_format
                        date_format = yellow_date_format
                    else:
                        continue  # skip non-highlighted rows

                    for col_idx, col_name in enumerate(contract_output.columns):
                        value = row[col_name]

                        if col_name in date_cols and pd.notna(value):
                            worksheet.write_datetime(row_number, col_idx, pd.to_datetime(value), date_format)
                        else:
                            worksheet.write(row_number, col_idx, value, base_format)
        
        else:
            # if no issue, drop the helper column (column with highlight) and write to file
            contract_output = contract_output.drop(columns=columns_with_highlight, errors='ignore')
            with pd.ExcelWriter(single_filepath, engine='xlsxwriter', datetime_format='m/d/yyyy') as writer:
                contract_output.to_excel(writer, sheet_name='Single Contract Upload', index=False)

        
    except Exception as e:
        error_msg = f"Error preparing single contract upload DataFrame: {str(e)}"
        print(error_msg)
        return False, error_msg
    
    return True, ""

def make_single_contract_csv(single_df, single_filepath_csv, contract_number):
    """Prepares a single contract csv by ensuring it has the correct columns and data types.
    the single_df can contain multiple contract, we will filter it by contract_number
    Args:
        single_df (pd.DataFrame): The DataFrame to prepare for single contract upload.
        single_filepath_csv: the path that temporarily stores the output file before zipping with other files
    Returns:
        bool: True if the DataFrame is successfully prepared, False otherwise.
        error_msg (str): Error message if preparation fails, empty string otherwise.
    """
    try: 
        if single_df.empty:
            # single_df can be empty, in this case we just write an empty file and return True
            single_df = pd.DataFrame(columns=['Mfg Part Num', 'Vendor Part Num',
                                            'Buyer Part Num', 'Description', 'Contract Price', 'UOM', 'QOE',
                                            'Effective Date', 'Expiration Date'])
            # write this to the single_filepath_csv and return true
            single_df.to_csv(single_filepath_csv, index=False)
            return True, ""
        
        # Filter by contract number
        contract_df = single_df[single_df['Contract Number (PrP)'] == contract_number].copy()
        if contract_df.empty:
            error_msg = f"No records found for contract number: {contract_number}"
            print(error_msg)
            return False, error_msg
        
        # format date columns
        date_columns = ['Effective Date', 'Expiration Date']
        for col in date_columns:
            if col in contract_df.columns:
                contract_df[col] = pd.to_datetime(contract_df[col], errors='coerce')
        
        # Ensure all required columns are present
        columns_no_highlight = [
            'Mfg Part Num', 'Vendor Part Num',
            'Buyer Part Num', 'Description', 'Contract Price', 'UOM', 'QOE',
            'Effective Date', 'Expiration Date', 
            'Actual Action', 
            'Final L Date Check', 'Final H Date Check', 'Create Count', 'ERP Vendor ID (PrP)'
        ]

        contract_columns = [col for col in columns_no_highlight if col in contract_df.columns]
        contract_output = contract_df[contract_columns].copy()

        # export as csv
        contract_output.to_csv(single_filepath_csv, index=False, date_format='%m/%d/%Y')
        return True, ""
    
    except Exception as e:
        error_msg = f"Error preparing single contract upload CSV: {str(e)}"
        print(error_msg)
        return False, error_msg


