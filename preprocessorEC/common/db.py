# common/db.py
from contextlib import contextmanager
from flask import current_app
import pandas as pd
from ..common.utils import reduce_mfg_part_num

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

def fix_encoding(text):
    """Fix text encoding issues and handle special characters safely"""
    if not isinstance(text, str):
        return text
    
    try:
        # First try direct normalization - this works for most cases
        import unicodedata
        normalized = unicodedata.normalize('NFKD', text)
        
        # Replace problematic characters
        char_map = {
            '\u201e': '"',  # double low-9 quotation mark
            '\u201c': '"',  # left double quotation mark
            '\u201d': '"',  # right double quotation mark
            '\u2018': "'",  # left single quotation mark
            '\u2019': "'",  # right single quotation mark
            '\u2013': '-',  # en dash
            '\u2014': '--', # em dash
        }
        
        for char, replacement in char_map.items():
            normalized = normalized.replace(char, replacement)
            
        return normalized
    except:
        # Fallback - just return the original text
        return text

def get_vendor_supplier_mapping(conn):
    """Get the vendor-supplier mapping from the database"""
    try:
        cursor = conn.cursor()
        query = f"""select Supplier, Vendor, VendorName
                from [DM_MONTYNT\\dli2].MDM_SUPPLIER_NAME_INFOR
                where active = 'Yes' AND VENDOR <> ''"""
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        return results
    except Exception as e:
        print(f"Database error: {str(e)}")
        return None
    
def get_valid_vid_in_list(conn):
    """Get the valid vendor IDs from the database
    return list of unique vendor IDs"""
    try:
        cursor = conn.cursor()
        query = f"""select distinct Vendor
                from [DM_MONTYNT\\dli2].MDM_SUPPLIER_NAME_INFOR
                where active = 'Yes' AND VENDOR <> ''"""
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
        # Extract unique vendor IDs from the results
        vendor_ids = set()
        for row in results:
            vendor_id = row['Vendor']
            if vendor_id:
                vendor_ids.add(vendor_id)
        return list(vendor_ids)
    except Exception as e: 
        print(f"Database error: {str(e)}")
        return None

def create_temp_table(table_name, df, conn):
    """Create a temporary table in the database for the uploaded file data"""
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
            Source_Contract_Type VARCHAR(20) NOT NULL,
            File_Row INT,
            PRIMARY KEY (Mfg_Part_Num, Contract_Number, UOM)
        )
        """
        cursor.execute(create_table_sql)
        
        # Define SQL table columns
        columns = ['Mfg_Part_Num', 'Vendor_Part_Num', 'Buyer_Part_Num', 'Description', 
                'Contract_Price', 'UOM', 'QOE', 'Effective_Date', 'Expiration_Date', 
                'Contract_Number', 'ERP_Vendor_ID', 'Reduced_Mfg_Part_Num', 'Source_Contract_Type',
                'File_Row']
        
        # columns we want from our input DataFrame
        pre_checked_columns = ['Mfg Part Num', 'Vendor Part Num', 'Buyer Part Num', 'Description',
                               'Contract Price', 'UOM', 'QOE', 'Effective Date', 'Expiration Date',
                               'Contract Number', 'ERP Vendor ID', 'Reduced Mfg Part Num', 'Source Contract Type',
                               'File Row']
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
            conn.commit()  # Commit each insert to avoid memory issues
        
        return True, table_name
    
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        print(f"Database error: {str(e)}")
        return False, str(e)

def drop_temp_table(table_name, conn):
    """Drop the temporary table from the database"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"IF OBJECT_ID('{table_name}', 'U') IS NOT NULL DROP TABLE {table_name}")
        return True, ""
    except Exception as e:
        print(f"Database error in drop_temp_table: {str(e)}")
        return False, str(e)

def find_duplicates_with_ccx(temp_table, conn):
    """Find potential duplicates between the temp table and CCX database"""
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
            try_convert(INT, ccx.QOE) AS qoe_ccx,
            try_convert(MONEY, ccx.PRICE) AS price_ccx,
            try_convert(DATE, ccx.ITEM_PRICE_START_DATE) AS effective_date_ccx,
            try_convert(DATE, ccx.ITEM_PRICE_END_DATE) AS expiration_date_ccx,
            ccx.VENDOR_PART_NUMBER AS vendor_part_num_ccx,
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
            temp.Source_Contract_Type,
            temp.Total_Contract_Line_Count,
            temp.File_Row,
            CASE WHEN ccx.MANUFACTURER_PART_NUMBER = temp.Mfg_Part_Num THEN 1 ELSE 0 END AS same_mfg_part_num
        FROM 
            (
                SELECT 
                    CONTRACT_NUMBER, 
                    CONTRACT_DESCRIPTION, 
                    CONTRACT_OWNER, 
                    IIF(SOURCE_CONTRACT_TYPE = 'GPO', 'GPO', 'Local') AS SOURCE_CONTRACT_TYPE,
                    CASE
                        WHEN MANUFACTURER_PART_NUMBER IS NULL OR LTRIM(RTRIM(MANUFACTURER_PART_NUMBER)) = '' THEN NULL
                        ELSE
                            CASE
                                WHEN TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(MANUFACTURER_PART_NUMBER)), '-', '')) IS NOT NULL
                                THEN CAST(TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(MANUFACTURER_PART_NUMBER)), '-', '')) AS VARCHAR(100))
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
                    IIF(VENDOR_PART_NUMBER = '<<N/A>>', '', VENDOR_PART_NUMBER) AS VENDOR_PART_NUMBER,
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
            (
                SELECT *,
                    count(1) over (partition by Contract_Number order by Contract_Number) as Total_Contract_Line_Count
                FROM
                    {temp_table}
            ) as temp
        ON 
            UPPER(CAST(ccx.REDUCED_MANUFACTURER_PART_NUMBER AS VARCHAR(100))) = UPPER(CAST(temp.Reduced_Mfg_Part_Num AS VARCHAR(100)))
        INNER JOIN
            (
             SELECT CONTRACT_NUMBER, Total_Line_Count
             FROM [DM_MONTYNT\\dli2].ccx_dump_line_count_stg
             ) [ccx_count]
        ON
            UPPER(ccx_count.CONTRACT_NUMBER) = UPPER(ccx.CONTRACT_NUMBER)
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
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        
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


def match_to_infor_contract_lines(temp_table, conn):
    """
    Match uploaded items to Infor Contract Lines
    
    Args:
        temp_table: Name of the temporary table containing upload data
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Execute the matching query
        match_query = f"""
            SELECT  
                infor.WorkingContractID AS contract_number_infor,
                infor.ContractName AS contract_description_infor,
                infor.ReducedManufacturerNumber AS reduced_mfg_part_num_infor,
                infor.ManufacturerName AS manufacturer_name_infor,
                infor.ManufacturerNumber AS mfg_part_num_infor,
                infor.UOM AS uom_infor,
                try_convert(INT, infor.DerivedUOMConversion) AS qoe_infor,
                try_convert(MONEY, infor.BaseCost) AS price_infor,
                try_convert(DATE, infor.EffectiveDate) AS effective_date_infor,
                try_convert(DATE, infor.ExpirationDate) AS expiration_date_infor,
                infor.VendorItem AS vendor_part_num_infor,
                infor.Vendor AS erp_vendor_id_infor,
                infor.VendorName AS vendor_name_infor,
                infor.ItemDescription AS description_infor,
                infor_count.ActiveLineCount AS total_line_count_infor,
                infor.ItemType AS item_type_infor,
                iif(infor.itemtype = 'Itemmast', infor.ItemNumber, '') AS item_number_infor,
                infor.[Contract] AS erp_contract_id_infor,
                infor.ContractLine AS erp_contract_line_infor,
                infor.Manufacturer AS erp_manufacturer_id_infor,
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
                temp.Source_Contract_Type,
                temp.Total_Contract_Line_Count,
                temp.File_Row,
                CASE WHEN infor.ManufacturerNumber = temp.Mfg_Part_Num THEN 1 ELSE 0 END AS same_mfg_part_num
            FROM 
                (
                    SELECT 
                        WorkingContractID, 
                        ContractName, 
                        Vendor,
                        VendorName,
                        cl.Manufacturer, 
                        mf.ManufacturerName,
                        ItemType,
                        ItemNumber,
                        VendorItem,
                        ManufacturerNumber,
                        CASE
                            WHEN ManufacturerNumber IS NULL OR LTRIM(RTRIM(ManufacturerNumber)) = '' THEN NULL
                            ELSE
                                CASE
                                    WHEN TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(ManufacturerNumber)), '-', '')) IS NOT NULL
                                    THEN CAST(TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(ManufacturerNumber)), '-', '')) AS VARCHAR(100))
                                    ELSE REPLACE(LTRIM(RTRIM(ManufacturerNumber)), '-', '')
                                END
                        END AS ReducedManufacturerNumber,
                        UOM,
                        BaseCost,
                        DerivedUOMConversion,
                        ItemDescription,
                        EffectiveDate,
                        ExpirationDate,
                        [Contract],
                        ContractLine
                    FROM 
                        [DM_MONTYNT\\dli2].CONTRACTLINE as cl
                    LEFT JOIN
                        [DM_MONTYNT\\dli2].MDM_MANUFACTURER_NAME_INFOR as mf
                    ON 
                        cl.Manufacturer = mf.Manufacturer
                    WHERE 
                        [Contract.ContractStatus] = 'Active'
                        AND [Contract.OnHold] = 'No'
                        AND ContractLineState = 'Active'
                        AND OnHold = 'No'
                        AND ActiveLine = 'Yes'
                        AND ExpirationDate >= getdate()
                ) as infor
            INNER JOIN 
                ( 
                SELECT *, 
                    count(1) over (partition by Contract_Number order by Contract_Number) as Total_Contract_Line_Count 
                FROM
                    {temp_table}
                ) as temp
            ON 
                upper(CAST(infor.ReducedManufacturerNumber AS VARCHAR(100))) = upper(CAST(temp.Reduced_Mfg_Part_Num AS VARCHAR(100)))
            INNER JOIN
                (
                 SELECT WorkingContractID, ActiveLineCount
                 FROM [DM_MONTYNT\\dli2].CONTRACTLINE_active_line_count_stg
                ) [infor_count]
            ON
                infor_count.WorkingContractID = infor.WorkingContractID
            ORDER BY 
                infor.WorkingContractID, infor.ManufacturerNumber
        """
        
        # Execute the query
        cursor.execute(match_query)
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        
        # Group by contract number (Infor side)
        contract_summary = {}
        for item in results:
            contract_num = item['contract_number_infor']
            if contract_num not in contract_summary:
                contract_summary[contract_num] = {
                    'contract_number': contract_num,
                    'contract_description': item['contract_description_infor'],
                    'manufacturer_name': item['manufacturer_name_infor'],
                    'total_matches': 0,
                    'exact_matches': 0,
                    'total_line_count_infor': item['total_line_count_infor'],
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
        error_msg = f"Error in match_to_infor_contract_lines: {str(e)}"
        current_app.logger.error(error_msg)
        if conn and 'conn' in locals() and not conn.closed:
            conn.close()
        return False, error_msg, None

def match_to_item_master(temp_table, conn):
    """
    Match uploaded items to Item Master data
    
    Args:
        temp_table: Name of the temporary table containing upload data
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Execute the matching query
        match_query = f"""
            SELECT DISTINCT
                vendoritem.Item as item_number_infor,
                vendoritem.ItemDescription as description_infor,
                vendoritem.Manufacturer as erp_manufacturer_id_infor,
                vendoritem.ManufacturerName as manufacturer_name_infor,
                vendoritem.ManufacturerNumber as mfg_part_num_infor,
                vendoritem.ReducedManufacturerNumber as reduced_mfg_part_num_infor,
                vendoritem.Vendor as erp_vendor_id_infor,
                vendoritem.VendorName as vendor_name_infor,
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
                temp.Source_Contract_Type,
                temp.File_Row,
                CASE WHEN vendoritem.ManufacturerNumber = temp.Mfg_Part_Num THEN 1 ELSE 0 END AS same_mfg_part_num
            FROM (
                SELECT 
                    Item, 
                    ItemDescription, 
                    Vendor, 
                    VendorName, 
                    VendorItem,
                    vi.Manufacturer, 
                    mf.ManufacturerName, 
                    ManufacturerNumber,
                    CASE
                        WHEN ManufacturerNumber IS NULL OR LTRIM(RTRIM(ManufacturerNumber)) = '' THEN NULL
                        ELSE
                            CASE
                                WHEN TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(ManufacturerNumber)), '-', '')) IS NOT NULL
                                THEN CAST(TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(ManufacturerNumber)), '-', '')) AS VARCHAR(100))
                                ELSE
                                    REPLACE(LTRIM(RTRIM(ManufacturerNumber)), '-', '')
                            END
                    END AS ReducedManufacturerNumber,
                    CASE
                        WHEN VendorItem IS NULL OR LTRIM(RTRIM(VendorItem)) = '' THEN NULL
                        ELSE
                            CASE
                                WHEN TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(VendorItem)), '-', '')) IS NOT NULL
                                THEN CAST(TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(VendorItem)), '-', '')) AS VARCHAR(100))
                                ELSE
                                    REPLACE(LTRIM(RTRIM(VendorItem)), '-', '')
                            END
                    END AS ReducedVendorNumber
                FROM [DM_MONTYNT\\dli2].MDM_VENDORITEM [vi]
                LEFT JOIN [DM_MONTYNT\\dli2].MDM_MANUFACTURER_NAME_INFOR [mf]
                ON vi.Manufacturer = mf.Manufacturer
                WHERE vi.Active = 'Yes'
            ) vendoritem
            INNER JOIN {temp_table} as temp
            ON upper(CAST(vendoritem.ReducedManufacturerNumber AS VARCHAR(100))) = upper(CAST(temp.Reduced_Mfg_Part_Num AS VARCHAR(100)))
            OR upper(CAST(vendoritem.ReducedVendorNumber AS VARCHAR(100))) = upper(CAST(temp.Reduced_Mfg_Part_Num AS VARCHAR(100)))
            ORDER BY 
                vendoritem.Item, vendoritem.ManufacturerNumber
        """
        
        # Execute the query
        cursor.execute(match_query)
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        
        # Group by item number (Item Master side)
        item_list = {}
        item_list['items'] = results
        item_list['total_matches'] = len(results)
        item_list['unique_matches'] = len(set(item['item_number_infor'] for item in results))
        item_list['false_positive_count'] = 0
        
        return True, "", item_list
        
    except Exception as e:
        error_msg = f"Error in match_to_item_master: {str(e)}"
        current_app.logger.error(error_msg)
        if conn and 'conn' in locals() and not conn.closed:
            conn.close()
        return False, error_msg, None


def get_valid_buying_uoms(item_numbers, conn):
    """
    Get valid buying UOM for a list of item numbers
    
    Args:
        item_numbers: List of item numbers
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Convert item numbers to a comma-separated string
        item_numbers_str = ', '.join(f"'{item}'" for item in item_numbers)
        
        # Execute the query
        query = f"""
            SELECT 
                    Item, 
                    UOM, 
                    UOMConversion, 
                    ValidForBuying, 
                    ItemDescription
                FROM 
                    [DM_MONTYNT\\dli2].MDM_ITEMUOM
                WHERE 
                    ([Item.Active] = 'Yes' AND ValidForBuying <> 'Not Valid') 
                    AND Item IN ({item_numbers_str})
        """
        
        cursor.execute(query)
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        
        return True, "", results
        
    except Exception as e:
        error_msg = f"Error in get_valid_buying_uom: {str(e)}"
        current_app.logger.error(error_msg)
        if conn and 'conn' in locals() and not conn.closed:
            conn.close()
        return False, error_msg, None



def get_relevant_contract_line(contract_numbers, conn):
    """
    Get relevant contract line details for a given contract number
    
    Args:
        contract_number: The list of contract numbers to search for
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()

        contract_numbers = ', '.join(f"'{contract}'" for contract in contract_numbers)
        
        # Execute the query
        query = f"""
            SELECT 
                contract_number,
                total_line_count
            FROM 
                [DM_MONTYNT\\dli2].ccx_dump_line_count_stg
            WHERE 
                contract_number IN ({contract_numbers})
        """
        
        cursor.execute(query)
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        
        return True, "", results
        
    except Exception as e:
        error_msg = f"Error in get_relevant_contract_line: {str(e)}"
        current_app.logger.error(error_msg)
        if conn and 'conn' in locals() and not conn.closed:
            conn.close()
        return False, error_msg, None


def commit_header(task_id, 
                  user_id, 
                  conn,
                  filename = None, 
                  precheck_mode = None, 
                  dedup_policy = None,
                  custom_direction = None,
                  custom_field = None,
                  simulation_mode = None,
                  with_error = None,
                  auto_complete = None):
    """
    Commit a new task header to the database
    
    Args:
        task_id: Unique identifier for the task
        user_id: ID of the user creating the task
        filename: Name of the file associated with the task
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        cursor = conn.cursor()
        insert_sql = """
            INSERT INTO [DM_MONTYNT\\dli2].PreprocessorHeader
            (TaskID, UserID, TPFileName, PreCheckMode,
            DedupMode, CustomDirection, CustomFields,
            SimulationMode, [Status], WithError,
            CreateDT, UpdateDT)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE(), GETDATE())
        """
        status = 'Pending' if auto_complete is False else 'Completed'
        cursor.execute(insert_sql, (task_id, user_id, filename, precheck_mode,
                                    dedup_policy, custom_direction, custom_field,
                                    simulation_mode, status, with_error))
        conn.commit()  # Commit the transaction
        
        return True, ""
        
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        error_msg = f"Error committing header: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg


def commit_all_changes(task_id, user_id, conn, final_all_changes=None):
    """Commit all changes for a given task ID and user ID"""
    try:
        cursor = conn.cursor()
        
        # Insert the final changes into the database
        if not final_all_changes or final_all_changes == []:
            error_msg = "No changes to commit"
            current_app.logger.error(error_msg)
            return True, error_msg
        
        for record in final_all_changes:
            # Extract and convert data types to ensure compatibility
            file_row = int(record.get('File Row', -1))
            dataset = str(record.get('Dataset', ''))
            contract_number = str(record.get('Contract Number', ''))
            mfg_part_num = str(record.get('Mfg Part Num', ''))
            vendor_part_num = str(record.get('Vendor Part Num', ''))
            buyer_part_num = str(record.get('Buyer Part Num', ''))
            description = str(record.get('Description', ''))
            
            # Handle numeric values
            try:
                contract_price = float(record.get('Contract Price', 0.0))
            except (TypeError, ValueError):
                contract_price = 0.0
                
            uom = str(record.get('UOM', ''))
            
            try:
                qoe = int(record.get('QOE', 0))
            except (TypeError, ValueError):
                qoe = 0
                
            effective_date = record.get('Effective Date', '1900-01-01')
            expiration_date = record.get('Expiration Date', '1900-12-31')
            erp_vendor_id = str(record.get('ERP Vendor ID', ''))
            primary_action = str(record.get('Primary Action', ''))
            actual_action = str(record.get('Actual Action2', ''))
            intended_action = str(record.get('Intended Action', ''))
            group = str(record.get('Group', ''))
            item = str(record.get('Item', ''))
            row_validation_flag = str(record.get('Final Row Validation Flag', ''))
            row_action = str(record.get('Final Row Action', ''))
            
            # Ensure these are non-empty strings since they're NOT NULL in the database
            file_row_validation_flag = str(record.get('Validation Flag', ''))
            if not file_row_validation_flag:
                file_row_validation_flag = "None"  # Use a placeholder string instead of empty string
                
            file_row_action = str(record.get('Final File Row Action', ''))
            if not file_row_action:
                file_row_action = "None"  # Use a placeholder string instead of empty string
            
            mfg_part_num_original = str(record.get('Mfg Part Num (Original)', ''))
            uom_original = str(record.get('UOM (Original)', ''))

            # Prepare the insert statement with exact column names matching the schema
            insert_sql = """
                INSERT INTO [DM_MONTYNT\\dli2].PreprocessorProcessedRaw
                (TaskID, UserID, CreateDT, UpdateDT, FileRow, DataSet, [Contract Number],
                [Mfg Part Num], [Vendor Part Num], [Buyer Part Num], [Description],
                [Contract Price], [UOM], [QOE], [Effective Date], [Expiration Date],
                [ERP Vendor ID], [Primary Action], [Actual Action], [Intended Action],
                [Group], [Item], [Row Validation Flag], [Row Action],
                [File Row Validation Flag], [File Row Action],
                [Mfg Part Num (Original)], [UOM (Original)])
                VALUES (?, ?, GETDATE(), GETDATE(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            cursor.execute(insert_sql, (task_id, user_id, file_row, dataset, contract_number,
                                       mfg_part_num, vendor_part_num, buyer_part_num,
                                       description, contract_price, uom, qoe,
                                       effective_date, expiration_date, erp_vendor_id,
                                       primary_action, actual_action, intended_action,
                                       group, item, row_validation_flag, row_action,
                                       file_row_validation_flag, file_row_action,
                                       mfg_part_num_original, uom_original))
            conn.commit()  # Commit the transaction
        return True, ""
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        error_msg = f"Error committing all changes: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg
    

def commit_commit_res(task_id, user_id, conn, final_commit_res=None):
    """Commit all changes for a given task ID and user ID"""
    try:
        cursor = conn.cursor()
        
        # Insert the final changes into the database
        if not final_commit_res or final_commit_res == []:
            error_msg = "No commit line to commit"
            current_app.logger.error(error_msg)
            return True, error_msg
        
        for record in final_commit_res:
            # Extract and convert data types to ensure compatibility
            file_row = int(record.get('File Row', -1))
            dataset = str(record.get('Dataset', ''))
            contract_number = str(record.get('Contract Number', ''))
            mfg_part_num = str(record.get('Mfg Part Num', ''))
            vendor_part_num = str(record.get('Vendor Part Num', ''))
            buyer_part_num = str(record.get('Buyer Part Num', ''))
            description = str(record.get('Description', ''))
            
            # Handle numeric values
            try:
                contract_price = float(record.get('Contract Price', 0.0))
            except (TypeError, ValueError):
                contract_price = 0.0
                
            uom = str(record.get('UOM', ''))
            
            try:
                qoe = int(record.get('QOE', 0))
            except (TypeError, ValueError):
                qoe = 0
                
            effective_date = record.get('Effective Date', '1900-01-01')
            expiration_date = record.get('Expiration Date', '1900-12-31')
            erp_vendor_id = str(record.get('ERP Vendor ID', ''))
            primary_action = str(record.get('Primary Action', ''))
            actual_action = str(record.get('Actual Action', ''))
            intended_action = str(record.get('Intended Action', ''))
            group = str(record.get('Group', ''))
            item = str(record.get('Item', ''))
            row_validation_flag = str(record.get('Final Row Validation Flag', ''))
            row_action = str(record.get('Final Row Action', ''))
            
            # Ensure these are non-empty strings since they're NOT NULL in the database
            file_row_validation_flag = str(record.get('Validation Flag', ''))
            if not file_row_validation_flag:
                file_row_validation_flag = "None"  # Use a placeholder string instead of empty string
                
            file_row_action = str(record.get('Final File Row Action', ''))
            if not file_row_action:
                file_row_action = "None"  # Use a placeholder string instead of empty string

            mfg_part_num_original = str(record.get('Mfg Part Num (Original)', ''))
            uom_original = str(record.get('UOM (Original)', ''))

            # Prepare the insert statement with exact column names matching the schema
            insert_sql = """
                INSERT INTO [DM_MONTYNT\\dli2].PreprocessorCommitLine
                (TaskID, UserID, CreateDT, UpdateDT, FileRow, DataSet, [Contract Number],
                [Mfg Part Num], [Vendor Part Num], [Buyer Part Num], [Description],
                [Contract Price], [UOM], [QOE], [Effective Date], [Expiration Date],
                [ERP Vendor ID], [Primary Action], [Actual Action], [Intended Action],
                [Group], [Item], [Row Validation Flag], [Row Action],
                [File Row Validation Flag], [File Row Action],
                [Mfg Part Num (Original)], [UOM (Original)])
                VALUES (?, ?, GETDATE(), GETDATE(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            cursor.execute(insert_sql, (task_id, user_id, file_row, dataset, contract_number,
                                       mfg_part_num, vendor_part_num, buyer_part_num,
                                       description, contract_price, uom, qoe,
                                       effective_date, expiration_date, erp_vendor_id,
                                       primary_action, actual_action, intended_action,
                                       group, item, row_validation_flag, row_action,
                                       file_row_validation_flag, file_row_action,
                                       mfg_part_num_original, uom_original))
            conn.commit()  # Commit the transaction
        
        return True, ""
    
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        error_msg = f"Error committing commit line (action lines): {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg
    
def commit_contract_line_count(task_id, user_id, conn, final_line_operations=None):
    """Commit contract line count for a given task ID and user ID"""
    try:
        cursor = conn.cursor()
        
        # Insert the final contract line counts into the database
        if not final_line_operations or final_line_operations == []:
            error_msg = "No contract line count to commit"
            current_app.logger.error(error_msg)
            return True, error_msg
        
        for record in final_line_operations:
            # Extract and convert data types to ensure compatibility
            contract_number = str(record.get('Contract Number', ''))
            total_line_count_original = int(record.get('Total Contract Line Count', 0))
            create = int(record.get('Create', 0))
            expire = int(record.get('Expire', 0))
            etc_create = int(record.get('Expire then Create (Create)', 0))
            etc_expire = int(record.get('Expire then Create (Expire)', 0))
            no_change = int(record.get('No Change', 0))
            update_existing = int(record.get('Update (Existing)', 0))
            update_new = int(record.get('Update (New)', 0))
            delta = int(record.get('Delta', 0))
            insert_count = int(record.get('Insert_Count', 0))
            update_count = int(record.get('Update_Count', 0))
            delete_count = int(record.get('Delete_Count', 0))
            total_line_count_after = int(record.get('Total Contract Line Count (Change Applied)', 0))
            
            # Prepare the insert statement with exact column names matching the schema
            insert_sql = """
                INSERT INTO [DM_MONTYNT\\dli2].PreprocessorContractLineCount
                (TaskID, UserID, CreateDT, UpdateDT, [Contract Number], [Total Lines (Original)],
                [Create], [Expire], [ETC (Create)], [ETC (Expire)],
                [No Change], [Update (Existing)], [Update (New)], [Delta], [Insert Count],
                [Update Count], [Delete Count], [Total Lines (Change Applied)])
                VALUES (?, ?, GETDATE(), GETDATE(), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            cursor.execute(insert_sql, (task_id, user_id, contract_number, 
                                       total_line_count_original, create, expire, 
                                       etc_create, etc_expire, no_change, 
                                       update_existing, update_new, delta, 
                                       insert_count, update_count, delete_count, 
                                       total_line_count_after))
            conn.commit()  # Commit the transaction
        
        return True, ""
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        error_msg = f"Error committing contract line count: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg


def commit_wrike(task_id, user_id, conn, wrike_task_id=None):
    """link the Wrike task # to task id and user id"""
    try:
        cursor = conn.cursor()
        
        # Insert the Wrike task ID into the PreprocessorWrike table
        insert_sql = """
            INSERT INTO [DM_MONTYNT\\dli2].PreprocessorWrike
            (TaskID, UserID, WrikeID, CreateDT, UpdateDT)
            VALUES (?, ?, ?, GETDATE(), GETDATE())
        """
        cursor.execute(insert_sql, (task_id, user_id, wrike_task_id))
        conn.commit()  # Commit the transaction
        
        return True, ""
        
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        error_msg = f"Error committing Wrike task: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg


def delete_existing_commit(user_id, task_id, conn):
    """
    Delete existing commit lines for a given task ID and user ID
    
    Args:
        user_id: ID of the user performing the delete
        task_id: Unique identifier for the task
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        cursor = conn.cursor()
        
        # Delete existing commit for the given task ID
        for table in ['PreprocessorProcessedRaw', 
                      'PreprocessorCommitLine', 
                      'PreprocessorContractLineCount',
                      'PreprocessorHeader',
                      'PreprocessorWrike']:
        
            delete_sql = f"""
                DELETE FROM [DM_MONTYNT\\dli2].{table}
                WHERE TaskID = ? AND UserID = ?
            """
            cursor.execute(delete_sql, (task_id, user_id))
            conn.commit()  # Commit after each delete to ensure changes are saved
        
        return True, ""
        
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        error_msg = f"Error deleting existing commited result for {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg
    

def get_task_history(conn, user_id = None, user_role = None):
    """
    Get task history for the user
    
    Args:
        conn: Database connection
        user_role: Role of the user (optional)
        user_id: ID of the user (optional)
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Build the query based on user role
        if user_role == 'admin' or user_role == 'mdm':
            query = """
                SELECT h.TaskID, h.UserID, w.WrikeID, TPFileName, PreCheckMode, DedupMode,
                       CustomDirection, CustomFields, SimulationMode,
                       Status, CompletedBy, ExportedBy,
                       h.CreateDT, h.UpdateDT
                FROM [DM_MONTYNT\\dli2].PreprocessorHeader [h]
                LEFT JOIN [DM_MONTYNT\\dli2].PreprocessorWrike [w]
                ON h.TaskID = w.TaskID
                WHERE Status <> 'Deleted'
                ORDER BY UpdateDT DESC, createDT DESC, WrikeID
            """
        else:
            query = """
                SELECT h.TaskID, h.UserID, w.WrikeID, TPFileName, PreCheckMode, DedupMode,
                       CustomDirection, CustomFields, SimulationMode,
                       Status, CompletedBy, ExportedBy,
                       h.CreateDT, h.UpdateDT
                FROM [DM_MONTYNT\\dli2].PreprocessorHeader [h]
                LEFT JOIN [DM_MONTYNT\\dli2].PreprocessorWrike [w]
                ON h.TaskID = w.TaskID
                where h.UserID = ? AND
                Status <> 'Deleted'
                ORDER BY UpdateDT DESC, createDT DESC, WrikeID
            """
        
        # Execute the query
        if user_role == 'admin' or user_role == 'mdm':
            cursor.execute(query)
        else:
            cursor.execute(query, (user_id,))
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        
        return True, "", results
        
    except Exception as e:
        error_msg = f"Error getting task history: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None



def get_data_export_line(conn, user_id = None, user_role = None):
    """
    Get data export line for the user
    
    Args:
        conn: Database connection
        user_role: Role of the user (optional)
        user_id: ID of the user (optional)
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Build the query based on user role
        if user_role == 'admin' or user_role == 'mdm':
            query = """
                SELECT *
                FROM [DM_MONTYNT\\dli2].[vw_PreprocessorExportFinal]
            """
        else:
            query = """
                SELECT *
                FROM [DM_MONTYNT\\dli2].[vw_PreprocessorExportFinal]
                WHERE UserID = ?
            """
        
        # Execute the query
        if user_role == 'admin' or user_role == 'mdm':
            cursor.execute(query)
        else:
            cursor.execute(query, (user_id,))
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        
        return True, "", results
        
    except Exception as e:
        error_msg = f"Error getting data export line: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None

def get_im_link_line(conn, user_id = None, user_role = None):
    """
    Get item master link line for the user
    
    Args:
        conn: Database connection
        user_role: Role of the user (optional)
        user_id: ID of the user (optional)
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Build the query based on user role
        if user_role == 'admin' or user_role == 'mdm':
            query = """
                SELECT *
                FROM [DM_MONTYNT\\dli2].[vw_PreprocessorItemToLink]
            """
        else:
            query = """
                SELECT *
                FROM [DM_MONTYNT\\dli2].[vw_PreprocessorItemToLink]
                WHERE UserID = ?
            """
        
        # Execute the query
        if user_role == 'admin' or user_role == 'mdm':
            cursor.execute(query)
        else:
            cursor.execute(query, (user_id,))
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        
        return True, "", results
        
    except Exception as e:
        error_msg = f"Error getting item master link line: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None
    

def get_contract_to_close(conn, user_id = None, user_role = None):
    """
    Get contract lines to close for the user
    
    Args:
        conn: Database connection
        user_role: Role of the user (optional)
        user_id: ID of the user (optional)
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Build the query based on user role
        if user_role == 'admin' or user_role == 'mdm':
            query = """
                select *
                from [DM_MONTYNT\\dli2].vw_PreprocessorCloseContract
            """
        else:
            query = """
                select *
                from [DM_MONTYNT\\dli2].vw_PreprocessorCloseContract
            """
        
        # Execute the query
        # note this is the place we alllow sourcing to view other's work
        # if user A thinks they are trying to update something but find out here that
        # after consolidation the contract will get expired, then user may need to
        # get into touch with other user to try to resolve the issue
        if user_role == 'admin' or user_role == 'mdm':
            cursor.execute(query)
        else:
            cursor.execute(query)
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        
        return True, "", results
        
    except Exception as e:
        error_msg = f"Error getting contract to close: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None


def get_error_lines(conn, user_id = None, user_role = None):
    """
    Get error lines for the user
    
    Args:
        conn: Database connection
        user_role: Role of the user (optional)
        user_id: ID of the user (optional)
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Build the query based on user role
        if user_role == 'admin' or user_role == 'mdm':
            query = """
                SELECT *
                FROM [DM_MONTYNT\\dli2].[vw_PreprocessorErrorLines]
                where [File Row Action] = 'Pending'
            """
        else:
            query = """
                SELECT *
                FROM [DM_MONTYNT\\dli2].[vw_PreprocessorErrorLines]
                WHERE UserID = ?
                and [File Row Action] = 'Pending'
            """
        
        # Execute the query
        if user_role == 'admin' or user_role == 'mdm':
            cursor.execute(query)
        else:
            cursor.execute(query, (user_id,))
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            cleaned_row = [fix_encoding(item) for item in row]
            results.append(dict(zip(columns, cleaned_row)))
        
        return True, "", results
        
    except Exception as e:
        error_msg = f"Error getting error lines: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None


def data_persistence_after_export(conn, user_id=None, user_role=None, zip_filename=None):
    """
    Get data persistence after export for the user
    
    Args:
        conn: Database connection
        user_role: Role of the user (optional)
        user_id: ID of the user (optional)
        zip_filename: Name of the exported ZIP file
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Ensure only 'mdm' or 'admin' roles can call this function
        if user_role not in ['mdm', 'admin']:
            error_msg = "Unauthorized access: Only 'mdm' or 'admin' roles can call this function."
            current_app.logger.error(error_msg)
            return False, error_msg

        cursor = conn.cursor()

        # Stored procedure call with @ExportedBy parameter
        query = """
            EXEC [DM_MONTYNT\\dli2].[sp_ExportPreprocessorData] @ExportedBy = ?, @ZipFile = ?
        """

        # Execute the query
        cursor.execute(query, (user_id, zip_filename))
        conn.commit()  # Commit the transaction

        # If no exception is raised, the execution was successful
        return True, ""

    except Exception as e:
        # Capture and log the error message
        conn.rollback()
        error_msg = f"Error executing stored procedure: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg