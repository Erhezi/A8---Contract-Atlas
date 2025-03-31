from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, send_from_directory, jsonify, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import pandas as pd
from utils import validate_file
import os
import time
import re
from random import randint
import json
from contextlib import contextmanager

main_blueprint = Blueprint('main', __name__, template_folder='templates')

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

@main_blueprint.route('/process-duplicates', methods=['POST'])
@login_required
def process_duplicates():
    """Process the uploaded data to find duplications with CCX data"""
    try:
        # Check if we have validated data
        if 'validated_data' not in session:
            return jsonify({
                'success': False,
                'message': 'No validated data available. Please complete Step 1 first.'
            })
        
        # Convert validated data from session back to dataframe
        validated_df = pd.DataFrame.from_dict(session['validated_data'])
        
        # Get single connection for entire operation
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False,
                'message': 'Failed to connect to database'
            })
        
        try:
            # Create a unique table name
            random_suffix = ''.join([str(randint(0, 9)) for _ in range(4)])
            user_id = current_user.id
            table_name = f"##temp_contract_{user_id}_{random_suffix}"  # Global Temp Table
            table_name = f"test_temp_contract_{user_id}" # for testing purpose - TEST TEST TEST
            print(table_name) # Debugging line - TEST TEST TEST
            
            # Create the temp table using the connection
            success, result = create_temp_table(table_name, validated_df, conn)
            
            if not success:
                return jsonify({
                    'success': False,
                    'message': f'Failed to create temporary table: {result}'
                })
            
            # Run duplicate checking using same connection
            success, error_msg, contract_list = find_duplicates_with_ccx(table_name, conn)
            
            if not success:
                return jsonify({
                    'success': False,
                    'message': f'Error finding duplicates: {error_msg}'
                })
            
            # Store results in session
            session['contract_duplicates'] = contract_list
            
            # Commit at the end of all operations
            conn.commit()
            
            return jsonify({
                'success': True,
                'message': 'Duplicate analysis complete',
                'contracts': contract_list
            })
        finally:
            # Always close connection
            conn.close()
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing duplicates: {str(e)}'
        })

@main_blueprint.route('/get-duplicate-contracts', methods=['GET'])
@login_required
def get_duplicate_contracts():
    """Get the list of contracts with potential duplicates"""
    if 'contract_duplicates' not in session:
        # If not in session yet, process the duplicates
        return jsonify({
            'success': False,
            'message': 'No duplicate data available. Please process duplicates first.'
        })
    
    return jsonify({
        'success': True,
        'contracts': session['contract_duplicates']
    })

@main_blueprint.route('/get-duplicate-items/<contract_num>', methods=['GET'])
@login_required
def get_duplicate_items(contract_num):
    """Get the items for a specific contract"""
    if 'contract_duplicates' not in session:
        return jsonify({
            'success': False,
            'message': 'No duplicate data available. Please process duplicates first.'
        })
    
    # Find the contract in the session data
    contract_list = session['contract_duplicates']
    contract = next((c for c in contract_list if c['contract_number'] == contract_num), None)
    
    if not contract:
        return jsonify({
            'success': False,
            'message': f'Contract {contract_num} not found'
        })
    
    return jsonify({
        'success': True,
        'contract': contract
    })

@main_blueprint.route('/include-exclude-contract', methods=['POST'])
@login_required
def include_exclude_contract():
    """Include or exclude a contract from further processing"""
    data = request.json
    contract_num = data.get('contract_num')
    include = data.get('include', True)
    
    if not contract_num:
        return jsonify({
            'success': False,
            'message': 'No contract number provided'
        })
    
    if 'contract_duplicates' not in session:
        return jsonify({
            'success': False,
            'message': 'No duplicate data available. Please process duplicates first.'
        })
    
    # Initialize included/excluded contracts if not present
    if 'included_contracts' not in session:
        session['included_contracts'] = [contract['contract_number'] for contract in contract_num]
    if 'excluded_contracts' not in session:
        session['excluded_contracts'] = []
    
    # Update the lists
    included = session['included_contracts']
    excluded = session['excluded_contracts']
    
    if include:
        if contract_num not in included:
            included.append(contract_num)
        if contract_num in excluded:
            excluded.remove(contract_num)
    else:
        if contract_num not in excluded:
            excluded.append(contract_num)
        if contract_num in included:
            included.remove(contract_num)
    
    # Update session
    session['included_contracts'] = included
    session['excluded_contracts'] = excluded
    
    return jsonify({
        'success': True,
        'included_contracts': included,
        'excluded_contracts': excluded
    })

@main_blueprint.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    return redirect(url_for('auth.landing'))

@main_blueprint.route('/index')
@login_required
def index():
    return render_template('index.html')

@main_blueprint.route('/dashboard')
@login_required
def dashboard():
    # Check if we need to restart the process
    if request.args.get('restart'):
        session['current_step_id'] = 1
        session['completed_steps'] = []
        flash('Starting a new process', 'info')
    
    # Get current step and all steps
    current_step = current_app.get_current_step()
    all_steps = current_app.get_all_steps()
    
    return render_template('dashboard.html', current_step=current_step, steps=all_steps)

@main_blueprint.route('/goto-step/<int:step_id>', methods=['GET'])
@login_required
def goto_step(step_id):
    """Navigate to a specific step"""
    # Check if the requested step is accessible
    is_allowed, current_step_id, message = current_app.validate_step_progress(step_id)
    
    if is_allowed:
        # Update the current step in the session without affecting completion status
        session['current_step_id'] = step_id
        return redirect(url_for('main.dashboard'))
    else:
        flash(message, 'warning')
        return redirect(url_for('main.dashboard'))

@main_blueprint.route('/step/<int:step_id>')
@login_required
def step_view(step_id):
    # Validate step access
    is_allowed, current_id, message = current_app.validate_step_progress(step_id)
    
    if not is_allowed:
        flash(message, 'danger')
        return redirect(url_for('main.dashboard'))
    
    step_id = max(1, min(step_id, 8))  # Clamp between 1 and 8
    session['viewed_step_id'] = step_id  # Track which step is being viewed
    
    return render_template('dashboard.html')

@main_blueprint.route('/process-step/<int:step_id>', methods=['POST'])
@login_required
def process_step(step_id):
    # Validate if user can process this step
    is_allowed, current_id, message = current_app.validate_step_progress(step_id)
    
    if not is_allowed:
        flash(message, 'danger')
        return redirect(url_for('main.dashboard'))
    
    # Process step based on its ID
    success = False
    error_msg = None
    
    try:
        if step_id == 1:
            # Step 1: File validation
            # Check if we have validated data in the session
            if 'validated_data' not in session or 'column_mapping' not in session:
                raise ValueError("File validation incomplete. Please upload and validate a file first.")
            
            # Get validated data from session
            validated_data = session.get('validated_data')
            
            # If we have validated data, then the file has been processed successfully
            if validated_data:
                success = True
                flash("File validated successfully!", "success")
            else:
                raise ValueError("File validation failed or no validated data available.")
            
        elif step_id == 2:
            # Duplication overview logic
            if 'validated_data' not in session:
                raise ValueError("No validated data available. Please complete Step 1 first.")
            
            # Convert validated data from session back to dataframe
            validated_df = pd.DataFrame.from_dict(session['validated_data'])
            
            # Get column mapping from session
            column_mapping = session.get('column_mapping', {})
            
            # Check for duplicates based on key fields
            # Typically, duplicates would be identified based on vendor part number, manufacturer part number, contract number
            duplicate_keys = ['Mfg Part Num', 'Vendor Part Num', 'Contract Number']
            
            # Map the duplicate keys to the actual column names in the dataframe
            mapped_dup_keys = [column_mapping[key] for key in duplicate_keys if key in column_mapping]
            
            if not mapped_dup_keys:
                raise ValueError("Required fields for duplication check not found in column mapping")
            
            # Find duplicates
            duplicates = validated_df.duplicated(subset=mapped_dup_keys, keep=False)
            
            if duplicates.any():
                # Get duplicate records
                duplicate_df = validated_df[duplicates].copy()
                
                # Add a row index for reference (Excel-style row numbering)
                duplicate_df['Excel Row'] = duplicate_df.index + 2  # +2 for header row + 1-based indexing
                
                # Reorder columns for better visibility
                cols = ['Excel Row'] + [col for col in duplicate_df.columns if col != 'Excel Row']
                duplicate_df = duplicate_df[cols]
                
                # Store duplicate info in session (for use in step 3)
                session['duplicates'] = {
                    'count': int(duplicates.sum()),
                    'keys': duplicate_keys,
                    'duplicate_indices': duplicates[duplicates].index.tolist()
                }
                
                # Generate HTML table of duplicates for display
                duplicate_table = duplicate_df.to_html(classes='table table-striped table-bordered', index=False)
                session['duplicate_table'] = duplicate_table
                
                # We still consider this step successful as we've identified duplicates
                success = True
                flash(f"Found {duplicates.sum()} duplicate entries. Please review in Step 3.", "warning")
            else:
                # No duplicates found
                session['duplicates'] = {'count': 0, 'keys': duplicate_keys, 'duplicate_indices': []}
                session['duplicate_table'] = None
                success = True
                flash("No duplicates found in the data.", "success")
            
        elif step_id == 3:
            # Duplication resolution logic
            resolution_strategy = request.form.get('resolution_strategy')
            if not resolution_strategy:
                raise ValueError("No resolution strategy selected")
                
            # Process resolution (dummy implementation)
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 4:
            # Item master matching logic
            matching_method = request.form.get('matching_method')
            if not matching_method:
                raise ValueError("No matching method selected")
                
            # Process matching (dummy implementation)
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 5:
            # Change simulation logic
            # Process simulation approval
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 6:
            # Export changes logic
            export_format = request.form.get('export_format')
            if not export_format:
                raise ValueError("No export format selected")
                
            # Process export (dummy implementation)
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 7:
            # Synchronization inspection logic
            # Process synchronization verification
            time.sleep(1)  # Simulate processing time
            success = True
            
        elif step_id == 8:
            # Completion verification - no processing needed
            success = True
            
        # If successful, mark step as complete and advance
        if success:
            next_step = current_app.mark_step_complete(step_id)
            flash(f"Step {step_id} completed successfully!", "success")
            return redirect(url_for('main.dashboard'))
            
    except ValueError as e:
        error_msg = str(e)
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        
    if error_msg:
        flash(error_msg, "danger")
        
    return redirect(url_for('main.dashboard'))

@main_blueprint.route('/map-columns', methods=['POST'])
@login_required
def map_columns():
    """Save column mapping to session"""
    if 'file_info' not in session:
        return jsonify({'success': False, 'message': 'No file uploaded. Please upload a file first.'})
    
    # Get column mapping from POST data
    column_mapping = {}
    for field, column in request.form.items():
        if column:  # Only include fields that are mapped to a column
            column_mapping[field] = column
    
    # Validate required fields
    required_fields = ['Mfg Part Num', 'Vendor Part Num', 'Description', 
                       'Contract Price', 'UOM', 'QOE', 
                       'Effective Date', 'Expiration Date', 'Contract Number', 'ERP Vendor ID']
    
    missing_fields = [field for field in required_fields if field not in column_mapping and field != 'Buyer Part Num']
    
    if missing_fields:
        return jsonify({
            'success': False, 
            'message': f'Missing required field mapping: {", ".join(missing_fields)}'
        })
    
    # Store mapping in session
    session['column_mapping'] = column_mapping
    
    return jsonify({
        'success': True,
        'message': 'Column mapping saved successfully'
    })

@main_blueprint.route('/validate-file', methods=['POST'])
@login_required
def validate_file_route():
    """Validate the uploaded file columns against required fields"""
    if 'file_info' not in session:
        return jsonify({'success': False, 'message': 'No file uploaded. Please upload a file first.'})
    
    file_info = session['file_info']
    column_mapping = {}
    
    # Get column mapping from POST data
    for field, column in request.form.items():
        if column:  # Only include fields that are mapped to a column
            column_mapping[field] = column
    
    # Load the file
    file_path = os.path.join(current_app.static_folder, 'uploads', file_info['saved_name'])
    
    try:
        # Read file
        if file_info['type'] == 'csv':
            df = pd.read_csv(file_path, dtype=str)
        else:
            df = pd.read_excel(file_path, dtype=str)

        # Get duplicate checking mode from session or form
        duplicate_mode = request.form.get('duplicate_mode', session.get('duplicate_check_mode', 'default'))
        session['duplicate_check_mode'] = duplicate_mode
        
        # Validate the file
        valid_df, error_df, has_errors = validate_file(df, column_mapping, duplicate_mode)
        
        if isinstance(has_errors, str):
            # This means there was an error in the validation process
            return jsonify({'success': False, 'message': has_errors})
        
        # Save the full result dataframe for user download
        from utils import save_error_file
        
        if has_errors:
           # Calculate statistics
            total_rows = int(len(error_df))  # Convert to Python int
            error_rows = int(error_df['Has Error'].sum())  # Convert to Python int
            duplicate_rows = int(len(error_df[(error_df['Warning-Potential Duplicates'] != '')]))  # Convert to Python int
            
            # Save the full result dataframe for user download
            error_file = save_error_file(error_df, current_user.id, file_info['name'])
            session['error_file'] = error_file
            
            # Convert error dataframe to HTML table for display
            # First, get only the rows with errors
            error_rows_df = error_df[error_df['Has Error']].copy()

            # Move the File Row column to the front for better visibility
            cols = ['File Row'] + [col for col in error_rows_df.columns if col != 'File Row']
            error_rows_df = error_rows_df[cols]

            # Generate HTML table properly by stripping out pandas default attributes
            html_content = error_rows_df.to_html(classes="", index=False, header=True)
            # Replace the entire table tag, not just part of it
            html_content = re.sub(r'<table[^>]*>', '', html_content)
            html_content = html_content.replace('</table>', '')
            error_table = f'<div class="table-responsive-container"><table id="error_table" class="table table-striped table-bordered">{html_content}</table></div>'
            
            return jsonify({
                'success': False, 
                'message': 'Validation failed. See the errors below. Review and make changes accordingly, then try again.',
                'error_table': error_table,
                'stats': {
                    'total_rows': total_rows,
                    'error_rows': error_rows,
                    'duplicate_rows': duplicate_rows
                }
            })
                
        # No errors, store validated dataframe in session
        # Since we can't store the dataframe directly in session, convert to dict
        session.pop('error_file', None)  # Clear any previous error file
        total_rows = len(valid_df)
        session['validated_data'] = valid_df.to_dict()
        # session['result_file'] = save_error_file(valid_df, current_user.id, file_info['name'])
        
        return jsonify({
            'success': True,
            'message': 'File validation passed successfully!',
            'stats': {
                'total_rows': total_rows,
                'error_rows': 0,
                'duplicate_rows': 0
            },
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error validating file: {str(e)}'})



@main_blueprint.route('/download-error-file')
@login_required
def download_error_file():
    """Download the error file"""
    # Get the filename from the session
    if 'error_file' not in session:
        flash('No error file available for download', 'danger')
        return redirect(url_for('main.dashboard'))
    
    file_path = session['error_file']
    
    # Verify the file belongs to the current user
    expected_user_dir = f'user_{current_user.id}'
    if expected_user_dir not in file_path:
        flash('Access denied', 'danger')
        return redirect(url_for('main.dashboard'))
    
    # Get directory and filename
    directory = os.path.join(current_app.root_path, os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    
    # Return the file
    return send_from_directory(directory=directory, path=filename, as_attachment=True)

@main_blueprint.route('/download-result-file')
@login_required
def download_result_file():
    """Download the result file"""
    # Get the filename from the session
    if 'result_file' not in session:
        flash('No result file available for download', 'danger')
        return redirect(url_for('main.dashboard'))
    
    file_path = session['result_file']
    
    # Verify the file belongs to the current user
    expected_user_dir = f'user_{current_user.id}'
    if expected_user_dir not in file_path:
        flash('Access denied', 'danger')
        return redirect(url_for('main.dashboard'))
    
    # Get directory and filename
    directory = os.path.join(current_app.root_path, os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    
    # Return the file
    return send_from_directory(directory=directory, path=filename, as_attachment=True)

@main_blueprint.route('/upload-file', methods=['POST'])
@login_required
def upload_file():
    """Upload a file and return column headers"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    if file:
        # Ensure upload directory exists
        upload_dir = os.path.join(current_app.static_folder, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        saved_name = f"{int(time.time())}_{filename}"
        file_path = os.path.join(upload_dir, saved_name)
        file.save(file_path)
        
        # Determine file type
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        # Read headers
        try:
            if file_ext == 'csv':
                df = pd.read_csv(file_path, nrows=0)
            else:
                df = pd.read_excel(file_path, nrows=0)
                
            headers = df.columns.tolist()
            
            # Store file info in session
            session['file_info'] = {
                'name': filename,
                'saved_name': saved_name,
                'path': file_path,
                'type': file_ext
            }
            
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'filename': filename,
                'headers': headers
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error processing file: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'File upload failed'})


@main_blueprint.route('/get-duplicates', methods=['GET'])
@login_required
def get_duplicates():
    """Return duplicate data for display in the UI"""
    if 'duplicates' not in session or 'duplicate_table' not in session:
        return jsonify({
            'success': False,
            'message': 'No duplicate data available. Please complete Step 2 first.'
        })
    
    duplicates = session['duplicates']
    duplicate_table = session['duplicate_table']
    
    return jsonify({
        'success': True,
        'count': duplicates['count'],
        'keys': duplicates['keys'],
        'table': duplicate_table
    })


@main_blueprint.route('/download-template')
def download_template():
    """Download the upload template file"""
    try:
        # Use the same path structure as your UOM file in utils.py
        template_path = os.path.join(current_app.root_path, 'data', 'upload_template.xlsx')
        # Check if file exists
        if not os.path.exists(template_path):
            print(f"Template file not found at: {template_path}")
            flash("Template file not found. Please contact the administrator.", "error")
            return redirect(url_for('main.dashboard'))
        
        # Log that we found the file
        print(f"Serving template from: {template_path}")
        
        # Return the file as a download attachment
        return send_file(
            template_path,
            as_attachment=True,
            download_name='contract_price_template.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        # Log the exception
        print(f"Error downloading template: {str(e)}")
        flash(f"Error downloading template: {str(e)}", "error")
        return redirect(url_for('main.dashboard'))