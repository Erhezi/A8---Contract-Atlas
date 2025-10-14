# common/db.py
from contextlib import contextmanager
from flask import current_app
import pandas as pd
from ..common.utils import reduce_mfg_part_num
from flask import current_app


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


def get_contract_summary(conn, contract_number):
    """Fetch contract metadata for a specific contract number."""
    try:
        cursor = conn.cursor()
        query = (
            """
            SELECT 
                CONTRACT_NUMBER,
                CASE WHEN SOURCE_CONTRACT_TYPE = 'GPO' THEN 'GPO' ELSE 'Local' END AS SOURCE_CONTRACT_TYPE,
                MAX(ITEM_PRICE_END_DATE) AS CONTRACT_END_DATE,
                MAX(VENDOR_ERP_NUMBER) AS VENDOR_ERP_NUMBER
            FROM [DM_MONTYNT\\dli2].ccx_dump_validation_stg
            WHERE CONTRACT_NUMBER = ?
            GROUP BY CONTRACT_NUMBER,
                     CASE WHEN SOURCE_CONTRACT_TYPE = 'GPO' THEN 'GPO' ELSE 'Local' END
            """
        )
        cursor.execute(query, (contract_number,))
        row = cursor.fetchone()
        if not row:
            return None

        columns = [column[0] for column in cursor.description]
        cleaned_row = [fix_encoding(item) for item in row]
        result = dict(zip(columns, cleaned_row))
        return result
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
            Reduced_Vendor_Part_Num VARCHAR(255),
            Source_Contract_Type VARCHAR(20) NOT NULL,
            File_Row INT,
            PRIMARY KEY (Mfg_Part_Num, Contract_Number, UOM)
        )
        """
        cursor.execute(create_table_sql)
        
        # Define SQL table columns
        columns = ['Mfg_Part_Num', 'Vendor_Part_Num', 'Buyer_Part_Num', 'Description', 
                'Contract_Price', 'UOM', 'QOE', 'Effective_Date', 'Expiration_Date', 
                'Contract_Number', 'ERP_Vendor_ID', 'Reduced_Mfg_Part_Num', 'Reduced_Vendor_Part_Num',
                'Source_Contract_Type',
                'File_Row']
        
        # columns we want from our input DataFrame
        pre_checked_columns = ['Mfg Part Num', 'Vendor Part Num', 'Buyer Part Num', 'Description',
                               'Contract Price', 'UOM', 'QOE', 'Effective Date', 'Expiration Date',
                               'Contract Number', 'ERP Vendor ID', 'Reduced Mfg Part Num', 'Reduced Vendor Part Num',
                               'Source Contract Type',
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
                elif col == 'Reduced_Vendor_Part_Num' and 'Reduced Vendor Part Num' in df.columns:
                    insert_df[col] = df['Reduced Vendor Part Num']
                else:
                    print(f"Missing column in DataFrame: {col}")
                    insert_df[col] = None

        # Batch insert for better performance
        batch_size = 10000  # Adjust batch size as needed
        rows = [
            [None if pd.isnull(v) or v == '' else v for v in row.values]
            for _, row in insert_df.iterrows()
        ]

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            placeholders = ','.join(['?' for _ in columns])
            column_names = ','.join(columns)
            insert_sql = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"
            cursor.executemany(insert_sql, batch)
            print('Inserted batch from {} to {}'.format(i, i + len(batch) - 1)) # for monitoring process to get a sense of timing

        conn.commit()  # Commit once after all batches
        
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
        WITH ccx AS (
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
                            ELSE REPLACE(REPLACE(LTRIM(RTRIM(MANUFACTURER_PART_NUMBER)), '-', ''), '.', '')
                        END
                END AS REDUCED_MANUFACTURER_PART_NUMBER,
                CASE 
                    WHEN VENDOR_PART_NUMBER IS NULL OR LTRIM(RTRIM(VENDOR_PART_NUMBER)) = '' OR VENDOR_PART_NUMBER = '<<N/A>>' THEN NULL
                    ELSE
                        CASE
                            WHEN TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(VENDOR_PART_NUMBER)), '-', '')) IS NOT NULL
                            THEN CAST(TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(VENDOR_PART_NUMBER)), '-', '')) AS VARCHAR(100))
                            ELSE REPLACE(REPLACE(LTRIM(RTRIM(VENDOR_PART_NUMBER)), '-', ''), '.', '')
                        END
                END AS REDUCED_VENDOR_PART_NUMBER,
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
            FROM [DM_MONTYNT\\dli2].ccx_dump_validation_stg
            WHERE IS_SUOM = 'f'
            AND ITEM_PRICE_START_DATE <= LAST_UPDATE_DATE 
            AND ITEM_PRICE_END_DATE > LAST_UPDATE_DATE
        ),
        temp AS (
            SELECT *,
                COUNT(1) OVER (PARTITION BY Contract_Number ORDER BY Contract_Number) AS Total_Contract_Line_Count
            FROM {temp_table}
        ),
        ccx_count AS (
            SELECT CONTRACT_NUMBER, Total_Line_Count
            FROM [DM_MONTYNT\\dli2].ccx_dump_line_count_stg
        )
        SELECT DISTINCT *
        FROM (
            -- First JOIN on Manufacturer Part Number
            SELECT 
                ccx.CONTRACT_NUMBER AS contract_number_ccx,
                ccx.CONTRACT_DESCRIPTION AS contract_description_ccx,
                ccx.CONTRACT_OWNER AS contract_owner_ccx,
                ccx.SOURCE_CONTRACT_TYPE AS source_contract_type_ccx,
                ccx.REDUCED_MANUFACTURER_PART_NUMBER AS reduced_mfg_part_num_ccx,
                ccx.MANUFACTURER_NAME AS manufacturer_name_ccx,
                ccx.MANUFACTURER_PART_NUMBER AS mfg_part_num_ccx,
                ccx.UOM AS uom_ccx,
                TRY_CONVERT(INT, ccx.QOE) AS qoe_ccx,
                TRY_CONVERT(MONEY, ccx.PRICE) AS price_ccx,
                TRY_CONVERT(DATE, ccx.ITEM_PRICE_START_DATE) AS effective_date_ccx,
                TRY_CONVERT(DATE, ccx.ITEM_PRICE_END_DATE) AS expiration_date_ccx,
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
            FROM ccx
            INNER JOIN temp ON UPPER(ccx.REDUCED_MANUFACTURER_PART_NUMBER) = UPPER(temp.Reduced_Mfg_Part_Num)
            INNER JOIN ccx_count ON UPPER(ccx_count.CONTRACT_NUMBER) = UPPER(ccx.CONTRACT_NUMBER)

            UNION ALL

            -- Second JOIN on Vendor Part Number
            SELECT 
                ccx.CONTRACT_NUMBER AS contract_number_ccx,
                ccx.CONTRACT_DESCRIPTION AS contract_description_ccx,
                ccx.CONTRACT_OWNER AS contract_owner_ccx,
                ccx.SOURCE_CONTRACT_TYPE AS source_contract_type_ccx,
                ccx.REDUCED_MANUFACTURER_PART_NUMBER AS reduced_mfg_part_num_ccx,
                ccx.MANUFACTURER_NAME AS manufacturer_name_ccx,
                ccx.MANUFACTURER_PART_NUMBER AS mfg_part_num_ccx,
                ccx.UOM AS uom_ccx,
                TRY_CONVERT(INT, ccx.QOE) AS qoe_ccx,
                TRY_CONVERT(MONEY, ccx.PRICE) AS price_ccx,
                TRY_CONVERT(DATE, ccx.ITEM_PRICE_START_DATE) AS effective_date_ccx,
                TRY_CONVERT(DATE, ccx.ITEM_PRICE_END_DATE) AS expiration_date_ccx,
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
            FROM ccx
            INNER JOIN temp ON UPPER(ccx.REDUCED_VENDOR_PART_NUMBER) = UPPER(temp.Reduced_Vendor_Part_Num)
            INNER JOIN ccx_count ON UPPER(ccx_count.CONTRACT_NUMBER) = UPPER(ccx.CONTRACT_NUMBER)
        ) AS combined
        ORDER BY contract_number_ccx, mfg_part_num_ccx;
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
            WITH infor AS (
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
                            ELSE REPLACE(REPLACE(LTRIM(RTRIM(ManufacturerNumber)), '-', ''), '.', '')
                        END
                END AS ReducedManufacturerNumber,
                CASE
                    WHEN VendorItem IS NULL OR LTRIM(RTRIM(VendorItem)) = '' THEN NULL
                    ELSE
                        CASE
                            WHEN TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(VendorItem)), '-', '')) IS NOT NULL
                            THEN CAST(TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(VendorItem)), '-', '')) AS VARCHAR(100))
                            ELSE REPLACE(REPLACE(LTRIM(RTRIM(VendorItem)), '-', ''), '.', '')
                        END
                END AS ReducedVendorNumber,
                UOM,
                BaseCost,
                DerivedUOMConversion,
                ItemDescription,
                EffectiveDate,
                ExpirationDate,
                [Contract],
                ContractLine
            FROM [DM_MONTYNT\\dli2].CONTRACTLINE AS cl
            LEFT JOIN [DM_MONTYNT\\dli2].MDM_MANUFACTURER_NAME_INFOR AS mf
                ON cl.Manufacturer = mf.Manufacturer
            WHERE 
                [Contract.ContractStatus] = 'Active'
                AND [Contract.OnHold] = 'No'
                AND ContractLineState = 'Active'
                AND OnHold = 'No'
                AND ActiveLine = 'Yes'
                AND ExpirationDate >= GETDATE()
        ),
        temp AS (
            SELECT *, 
                COUNT(1) OVER (PARTITION BY Contract_Number ORDER BY Contract_Number) AS Total_Contract_Line_Count
            FROM {temp_table}   
        ),
        infor_count AS (
            SELECT WorkingContractID, ActiveLineCount
            FROM [DM_MONTYNT\\dli2].CONTRACTLINE_active_line_count_stg
        )
        SELECT DISTINCT *
        FROM (
            -- First join on Manufacturer Part Number
            SELECT  
                infor.WorkingContractID AS contract_number_infor,
                infor.ContractName AS contract_description_infor,
                infor.ReducedManufacturerNumber AS reduced_mfg_part_num_infor,
                infor.ManufacturerName AS manufacturer_name_infor,
                infor.ManufacturerNumber AS mfg_part_num_infor,
                infor.UOM AS uom_infor,
                TRY_CONVERT(INT, infor.DerivedUOMConversion) AS qoe_infor,
                TRY_CONVERT(MONEY, infor.BaseCost) AS price_infor,
                TRY_CONVERT(DATE, infor.EffectiveDate) AS effective_date_infor,
                TRY_CONVERT(DATE, infor.ExpirationDate) AS expiration_date_infor,
                infor.VendorItem AS vendor_part_num_infor,
                infor.Vendor AS erp_vendor_id_infor,
                infor.VendorName AS vendor_name_infor,
                infor.ItemDescription AS description_infor,
                infor_count.ActiveLineCount AS total_line_count_infor,
                infor.ItemType AS item_type_infor,
                IIF(infor.ItemType = 'Itemmast', infor.ItemNumber, '') AS item_number_infor,
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
            FROM infor
            INNER JOIN temp ON UPPER(infor.ReducedManufacturerNumber) = UPPER(temp.Reduced_Mfg_Part_Num)
            INNER JOIN infor_count ON infor_count.WorkingContractID = infor.WorkingContractID

            UNION ALL

            -- Second join on Vendor Part Number
            SELECT  
                infor.WorkingContractID AS contract_number_infor,
                infor.ContractName AS contract_description_infor,
                infor.ReducedManufacturerNumber AS reduced_mfg_part_num_infor,
                infor.ManufacturerName AS manufacturer_name_infor,
                infor.ManufacturerNumber AS mfg_part_num_infor,
                infor.UOM AS uom_infor,
                TRY_CONVERT(INT, infor.DerivedUOMConversion) AS qoe_infor,
                TRY_CONVERT(MONEY, infor.BaseCost) AS price_infor,
                TRY_CONVERT(DATE, infor.EffectiveDate) AS effective_date_infor,
                TRY_CONVERT(DATE, infor.ExpirationDate) AS expiration_date_infor,
                infor.VendorItem AS vendor_part_num_infor,
                infor.Vendor AS erp_vendor_id_infor,
                infor.VendorName AS vendor_name_infor,
                infor.ItemDescription AS description_infor,
                infor_count.ActiveLineCount AS total_line_count_infor,
                infor.ItemType AS item_type_infor,
                IIF(infor.ItemType = 'Itemmast', infor.ItemNumber, '') AS item_number_infor,
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
            FROM infor
            INNER JOIN temp ON UPPER(infor.ReducedVendorNumber) = UPPER(temp.Reduced_Vendor_Part_Num)
            INNER JOIN infor_count ON infor_count.WorkingContractID = infor.WorkingContractID
        ) AS combined
        ORDER BY contract_number_infor, mfg_part_num_infor;
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
                                    REPLACE(REPLACE(LTRIM(RTRIM(ManufacturerNumber)), '-', ''), '.', '')
                            END
                    END AS ReducedManufacturerNumber,
                    CASE
                        WHEN VendorItem IS NULL OR LTRIM(RTRIM(VendorItem)) = '' THEN NULL
                        ELSE
                            CASE
                                WHEN TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(VendorItem)), '-', '')) IS NOT NULL
                                THEN CAST(TRY_CONVERT(BIGINT, REPLACE(LTRIM(RTRIM(VendorItem)), '-', '')) AS VARCHAR(100))
                                ELSE
                                    REPLACE(REPLACE(LTRIM(RTRIM(VendorItem)), '-', ''), '.', '')
                            END
                    END AS ReducedVendorNumber
                FROM [DM_MONTYNT\\dli2].MDM_VENDORITEM [vi]
                LEFT JOIN [DM_MONTYNT\\dli2].MDM_MANUFACTURER_NAME_INFOR [mf]
                ON vi.Manufacturer = mf.Manufacturer
                WHERE vi.Active = 'Yes'
            ) vendoritem
            INNER JOIN {temp_table} as temp
            ON upper(CAST(vendoritem.ReducedManufacturerNumber AS VARCHAR(100))) = upper(CAST(temp.Reduced_Mfg_Part_Num AS VARCHAR(100)))
            OR upper(CAST(vendoritem.ReducedVendorNumber AS VARCHAR(100))) = upper(CAST(temp.Reduced_Vendor_Part_Num AS VARCHAR(100)))
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


def get_EDI_sub_UOM(conn):
    """
    Get EDI sub UOMs from the database
    
    Args:
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Execute the query
        query = f"""
           select 
            a.UOM, 
            COALESCE(b.LawsonValue, A.UOM) AS UOM_EDI,
            count(1) over (partition by a.uom order by a.uom) as ck
        from
            (select distinct UOM 
            from [DM_MONTYNT\\dli2].ccx_dump_validation_stg) [a]
            left join [DM_MONTYNT\\dli2].MDM_EDI_SUB_UOM [b]
            on a.uom = b.ExternalValue
            order by ck, a.UOM
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
        error_msg = f"Error in get_EDI_sub_UOM: {str(e)}"
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


def delete_existing_commit(user_role, task_id, conn):
    """
    Delete existing commit task for a given task ID (Admin only function)
    
    Args:
        user_id: ID of the user performing the delete
        task_id: Unique identifier for the task
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        if user_role != 'admin':
            error_msg = "Only admin users can delete existing commits"
            current_app.logger.error(error_msg)
            return False, error_msg
        
        cursor = conn.cursor()
        
        # Delete existing commit for the given task ID
        for table in ['PreprocessorProcessedRaw', 
                      'PreprocessorCommitLine', 
                      'PreprocessorContractLineCount',
                      'PreprocessorHeader',
                      'PreprocessorWrike',
                      'PreprocessorExported',
                      'PreprocessorContractLink',
                      'PreprocessorContractClose',
                      'PreprocessorCompletionComment',
                      'PreprocessorErrorEdit',
                      'PreprocessorExported']:
        
            delete_sql = f"""
                DELETE FROM [DM_MONTYNT\\dli2].{table}
                WHERE TaskID = ?
            """
            cursor.execute(delete_sql, (task_id,))
            conn.commit()  # Commit after each delete to ensure changes are saved
        
        return True, ""
        
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        error_msg = f"Error deleting existing commited result for {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg

def clear_deleted_tasks(user_role, task_ids, conn):
    """
    Clear deleted tasks for a given list of task IDs (Admin only function)
    
    Args:
        user_role: Role of the user performing the clear
        task_ids: List of unique identifiers for the tasks
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        if user_role != 'admin':
            error_msg = "Only admin users can clear deleted tasks"
            current_app.logger.error(error_msg)
            return False, error_msg
        
        cursor = conn.cursor()
        
        # Delete tasks from PreprocessorHeader and related tables
        for task_id in task_ids:
            delete_sql = """
                DELETE FROM [DM_MONTYNT\\dli2].PreprocessorHeader
                WHERE TaskID = ? and Status = 'Deleted'
            """
            cursor.execute(delete_sql, (task_id,))
            
            # Clear related tables
            for table in ['PreprocessorWrike', 
                          'PreprocessorProcessedRaw', 
                          'PreprocessorCommitLine',
                          'PreprocessorContractLineCount',
                          'PreprocessorExported',
                          'PreprocessorContractLink',
                          'PreprocessorContractClose',
                          'PreprocessorCompletionComment',
                          'PreprocessorErrorEdit']:
                clear_sql = f"""
                    DELETE FROM [DM_MONTYNT\\dli2].{table}
                    WHERE TaskID = ?
                """
                cursor.execute(clear_sql, (task_id,))
        
        conn.commit()  # Commit the transactions
        
        return True, ""
        
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        error_msg = f"Error clearing deleted tasks: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg


def repending_existing_exported_task(user_role, task_id, conn):
    """
    Repend existing exported task for a given task ID (Admin only function)
    
    Args:
        user_id: ID of the user performing the repending
        task_id: Unique identifier for the task
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        if user_role != 'admin':
            error_msg = "Only admin users can reinstall pending for existing exported task"
            current_app.logger.error(error_msg)
            return False, error_msg
        
        cursor = conn.cursor()
        
        # Update status to 'Pending' for the given task ID in PreprocessorHeader if the current status is 'Exported'
        check_sql = """
            SELECT Status FROM [DM_MONTYNT\\dli2].PreprocessorHeader
            WHERE TaskID = ?
        """
        cursor.execute(check_sql, (task_id,))
        row = cursor.fetchone()
        if not row or row[0] not in ('Exported', 'Deleted'):
            error_msg = f"Task {task_id} is not in 'Exported' or 'Deleted' status, cannot reset to 'pending'."
            current_app.logger.error(error_msg)
            return False, error_msg
        
        update_sql = """
            UPDATE [DM_MONTYNT\\dli2].PreprocessorHeader
            SET Status = 'Pending', Status2 = 'Pending', UpdateDT = GETDATE()
            WHERE TaskID = ?
        """
        cursor.execute(update_sql, (task_id,))

        # clear exported table for the task, we have a few tables to clear
        for table in ['PreprocessorContractLink',
                      'PreprocessorExported',
                      'PreprocessorCompletionComment',
                      'PreprocessorErrorEdit']:
            clear_sql = f"""
                DELETE FROM [DM_MONTYNT\\dli2].{table}
                WHERE TaskID = ?
            """
            cursor.execute(clear_sql, (task_id,))


        conn.commit()  # Commit the transactions
        
        return True, ""
        
    except Exception as e:
        conn.rollback()  # Rollback in case of error
        error_msg = f"Error repending existing exported result for {task_id}: {str(e)}"
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
        if user_role == 'mdm' or user_role == "general":
            query = """
                SELECT h.TaskID, h.UserID, w.WrikeID, TPFileName, PreCheckMode, DedupMode,
                       CustomDirection, CustomFields, SimulationMode,
                       Status, Status2, WithError, CompletedBy, ExportedBy,
                       h.CreateDT, h.UpdateDT
                FROM [DM_MONTYNT\\dli2].PreprocessorHeader [h]
                LEFT JOIN [DM_MONTYNT\\dli2].PreprocessorWrike [w]
                ON h.TaskID = w.TaskID
                WHERE Status <> 'Deleted'
                ORDER BY UpdateDT DESC, createDT DESC, WrikeID
            """
        elif user_role == 'admin':
            query = """
                SELECT h.TaskID, h.UserID, w.WrikeID, TPFileName, PreCheckMode, DedupMode,
                       CustomDirection, CustomFields, SimulationMode,
                       Status, Status2, WithError, CompletedBy, ExportedBy,
                       h.CreateDT, h.UpdateDT
                FROM [DM_MONTYNT\\dli2].PreprocessorHeader [h]
                LEFT JOIN [DM_MONTYNT\\dli2].PreprocessorWrike [w]
                ON h.TaskID = w.TaskID
                ORDER BY UpdateDT DESC, createDT DESC, WrikeID
            """
        else:
            query = """
                SELECT h.TaskID, h.UserID, w.WrikeID, TPFileName, PreCheckMode, DedupMode,
                       CustomDirection, CustomFields, SimulationMode,
                       Status, Status2, WithError, CompletedBy, ExportedBy,
                       h.CreateDT, h.UpdateDT
                FROM [DM_MONTYNT\\dli2].PreprocessorHeader [h]
                LEFT JOIN [DM_MONTYNT\\dli2].PreprocessorWrike [w]
                ON h.TaskID = w.TaskID
                where h.UserID = ? AND
                Status <> 'Deleted'
                ORDER BY UpdateDT DESC, createDT DESC, WrikeID
            """
        
        # Execute the query
        if user_role == 'admin' or user_role == 'mdm' or user_role == "general":
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
    

def get_contracts_to_link(conn):
    """
    Get contracts that need linking from PreprocessorExported table
    
    Args:
        conn: Database connection
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Execute the query to find contracts needing linkage
        query = """
            SELECT DISTINCT 
                TaskID, 
                [Export Group], 
                [Contract Number (PrP)], 
                [ERP Vendor ID (PrP)],
                [Contract Number], 
                [ERP Vendor ID (CCX Sync)], 
                ExportedBy
            FROM [DM_MONTYNT\\dli2].PreprocessorExported
            WHERE TaskID IN (
                SELECT TaskID 
                FROM [DM_MONTYNT\\dli2].PreprocessorHeader 
                WHERE Status2 = 'Pending'
            )
            ORDER BY [Export Group] desc, [Contract Number]
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
        error_msg = f"Error getting contracts to link: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None


def commit_contract_link(conn, task_id, contract_number_prp, erp_vendor_id_prp, 
                         contract_number, erp_vendor_id_ccx, export_group, user_id):
    """
    Save contract link information to PreprocessorContractLink table
    
    Args:
        conn: Database connection
        task_id: TaskID to link
        contract_number_prp: Contract number from PrP
        erp_vendor_id_prp: ERP vendor ID from PrP
        contract_number: Contract number for CCX Sync
        erp_vendor_id_ccx: ERP vendor ID for CCX Sync
        export_group: Export group
        user_id: Current user ID
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        cursor = conn.cursor()
        
        # Check if record exists
        cursor.execute("""
            SELECT TaskID FROM PreprocessorContractLink
            WHERE TaskID = ? AND [Contract Number (PrP)] = ? AND [ERP Vendor ID (PrP)] = ?
        """, (task_id, contract_number_prp, erp_vendor_id_prp))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            cursor.execute("""
                UPDATE [DM_MONTYNT\\dli2].PreprocessorContractLink
                SET [Contract Number] = ?, 
                    [ERP Vendor ID (CCX Sync)] = ?,
                    LinkedBy = ?,
                    UpdateDT = GETDATE()
                WHERE TaskID = ? 
                    AND [Contract Number (PrP)] = ? 
                    AND [ERP Vendor ID (PrP)] = ?
            """, (contract_number, erp_vendor_id_ccx, user_id, 
                  task_id, contract_number_prp, erp_vendor_id_prp))
        else:
            # Insert new record
            cursor.execute("""
                INSERT INTO [DM_MONTYNT\\dli2].PreprocessorContractLink
                (TaskID, [Export Group], [Contract Number (PrP)], [ERP Vendor ID (PrP)], 
                 [Contract Number], [ERP Vendor ID (CCX Sync)], LinkedBy, CreateDT, UpdateDT)
                VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE(), GETDATE())
            """, (task_id, export_group, contract_number_prp, erp_vendor_id_prp,
                  contract_number, erp_vendor_id_ccx, user_id))
        
        # conn.commit()

        cursor.execute("""
            EXEC [DM_MONTYNT\\dli2].sp_UpdatePreprocessorExportedFromLink
        """)
        
        # Commit the changes made by the stored procedure
        conn.commit()

        return True, ""
        
    except Exception as e:
        conn.rollback()
        error_msg = f"Error committing contract link: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg

def hold_task_by_task_id(conn, task_id, user_id):
    """
    Mark a task as held (on hold) in the database.

    Args:
        conn: Database connection.
        task_id: ID of the task to hold.
        user_id: ID of the user performing the action.

    Returns:
        Tuple of (success, error_message).
    """
    try:
        cursor = conn.cursor()
        # Update task status to 'Hold'
        cursor.execute("""
            UPDATE [DM_MONTYNT\\dli2].PreprocessorHeader
            SET Status = 'Hold', UpdateDT = GETDATE()
            WHERE TaskID = ? AND Status = 'Pending'
        """, (task_id,))
        
        # Check if any rows were affected
        if cursor.rowcount == 0:
            return False, "Task not found or not in 'Pending' status."
            
        conn.commit()
        return True, ""
    except Exception as e:
        conn.rollback()
        error_msg = f"Error holding task {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg


def unhold_task_by_task_id(conn, task_id, user_id):
    """
    Unhold a task (change from 'Hold' to 'Pending') in the database.

    Args:
        conn: Database connection.
        task_id: ID of the task to unhold.
        user_id: ID of the user performing the action.

    Returns:
        Tuple of (success, error_message).
    """
    try:
        cursor = conn.cursor()
        # Update task status from 'Hold' to 'Pending'
        cursor.execute("""
            UPDATE [DM_MONTYNT\\dli2].PreprocessorHeader
            SET Status = 'Pending', UpdateDT = GETDATE()
            WHERE TaskID = ? AND Status = 'Hold'
        """, (task_id,))
        
        # Check if any rows were affected
        if cursor.rowcount == 0:
            return False, "Task not found or not in 'Hold' status."
            
        conn.commit()
        return True, ""
    except Exception as e:
        conn.rollback()
        error_msg = f"Error unholding task {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg
    

def delete_task_by_task_id(conn, task_id, user_id):
    """
    Mark a task as deleted in the database.

    Args:
        conn: Database connection.
        task_id: ID of the task to delete.
        user_id: ID of the user performing the delete.

    Returns:
        Tuple of (success, error_message).
    """
    try:
        cursor = conn.cursor()
        # Update task status to 'Deleted' and set DeletedBy
        cursor.execute("""
            UPDATE [DM_MONTYNT\\dli2].PreprocessorHeader
            SET Status = 'Deleted', UpdateDT = GETDATE(), DeletedBy = ?
            WHERE TaskID = ?
        """, (user_id, task_id))
        conn.commit()
        return True, ""
    except Exception as e:
        conn.rollback()
        error_msg = f"Error deleting task {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg


def get_task_by_task_id(conn, task_id):
    """
    Retrieve task details by task ID.

    Args:
        conn: Database connection.
        task_id: ID of the task to retrieve.

    Returns:
        Tuple of (success, error_message, task_data).
    """
    try:
        cursor = conn.cursor()
        # Fetch task details
        cursor.execute("""
            SELECT TaskID, UserID, Status, Status2, WithError, PreCheckMode, DedupMode, SimulationMode, 
                   CreateDT, UpdateDT, TPFileName
            FROM [DM_MONTYNT\\dli2].PreprocessorHeader
            WHERE TaskID = ?
        """, (task_id,))
        task = cursor.fetchone()
        if not task:
            return False, "Task not found.", None

        # Map the result to a dictionary
        columns = [column[0] for column in cursor.description]
        task_data = dict(zip(columns, task))
        return True, "", task_data
    except Exception as e:
        error_msg = f"Error retrieving task {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None


def mark_task_reprocess(conn, task_id):
    """Mark related rows to 'Reprocess' in both 
    PreprocessorProcessedRaw and PreprocessorCommitLine for a task
    after the task's error edits are exported as new TP file, then 
    changne the header withError from 'Yes' to 'Rpr', and update the UpdateDT.
    """
    
    try:
        cur = conn.cursor()
        # Update ProcessedRaw file-row level action where Pending
        cur.execute(
            """
            UPDATE [DM_MONTYNT\\dli2].PreprocessorProcessedRaw
            SET [File Row Action] = 'Reprocess'
            WHERE TaskID = ? AND [File Row Action] = 'Pending'
            """,
            (task_id,)
        )
        # Update row-level action where Pending
        cur.execute(
            """
            UPDATE [DM_MONTYNT\\dli2].PreprocessorProcessedRaw
            SET [Row Action] = 'Reprocess'
            WHERE TaskID = ? AND [Row Action] = 'Pending'
            """,
            (task_id,)
        )

        # Update CommitLine as well
        cur.execute(
            """
            UPDATE [DM_MONTYNT\\dli2].PreprocessorCommitLine
            SET [File Row Action] = 'Reprocess'
            WHERE TaskID = ? AND [File Row Action] = 'Pending'
            """,
            (task_id,)
        )
        cur.execute(
            """
            UPDATE [DM_MONTYNT\\dli2].PreprocessorCommitLine
            SET [Row Action] = 'Reprocess'
            WHERE TaskID = ? AND [Row Action] = 'Pending'
            """,
            (task_id,)
        )

        # Update Header withError from Yes to Rpr
        cur.execute(
            """
            UPDATE [DM_MONTYNT\\dli2].PreprocessorHeader
            SET WithError = 'Rpr',
                UpdateDT = GETDATE()
            WHERE TaskID = ?
            """,
            (task_id,)
        )

        conn.commit()
        return True, ""
    except Exception as e:
        conn.rollback()
        msg = f"Error marking task {task_id} reprocess: {str(e)}"
        print(msg)
        return False, msg


def apply_approved_edits(conn, task_id, approvals):
    """Apply approved edits: merge to PreprocessorCommitLine and mark actions/flags in both tables.
    approvals: list of {data: row_dict, message}
    Returns (success, msg, applied_count)
    """
    try:
        cur = conn.cursor()
        applied = 0
        pass_msg = 'PASS - safe to retain/drop'
        for ap in approvals:
            r = ap.get('data', {}) if isinstance(ap, dict) else {}
            file_row = r.get('FileRow')
            contract_number = r.get('Contract Number')

            # Update CommitLine actions to Execute and set validation flags
            cur.execute(
                """
                UPDATE [DM_MONTYNT\\dli2].PreprocessorCommitLine
                SET [File Row Action] = 'Execute',
                    [Row Action] = 'Execute',
                    [File Row Validation Flag] = ?,
                    [Row Validatation Flag] = ?
                WHERE TaskID = ? AND [FileRow] = ? AND [Contract Number] = ?
                  AND [File Row Action] = 'Pending'
                """,
                (pass_msg, pass_msg, task_id, file_row, contract_number)
            )

            # Mirror in ProcessedRaw
            cur.execute(
                """
                UPDATE [DM_MONTYNT\\dli2].PreprocessorProcessedRaw
                SET [File Row Action] = 'Execute',
                    [Row Action] = 'Execute'
                WHERE TaskID = ? AND [FileRow] = ? AND [Contract Number] = ?
                  AND ([File Row Action] = 'Pending' OR [Row Action] = 'Pending')
                """,
                (task_id, file_row, contract_number)
            )
            applied += cur.rowcount

        conn.commit()
        return True, "", applied
    except Exception as e:
        conn.rollback()
        msg = f"Error applying approved edits for {task_id}: {str(e)}"
        print(msg)
        return False, msg, 0


def get_task_errors(conn, task_id):
    """
    Retrieve error records for a given task ID.
    
    Args:
        conn: Database connection
        task_id: ID of the task to retrieve errors for
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # SQL query to fetch error records
        query = """
        (
        SELECT *
        FROM [DM_MONTYNT\\dli2].PreprocessorProcessedRaw
        WHERE TaskID = ?
          AND [File Row Action] = 'Pending'
          AND [Group] = 'Keep'

        UNION ALL

        SELECT *
        FROM [DM_MONTYNT\\dli2].PreprocessorProcessedRaw
        WHERE TaskID = ?
          AND [Row Action] = 'Pending'
        )
        ORDER BY [FileRow], [DataSet] DESC
        """
        
        # Execute query with parameters
        cursor.execute(query, (task_id, task_id))
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
        
        return True, "", results
        
    except Exception as e:
        error_msg = f"Error retrieving task errors: {str(e)}"
        print(error_msg)
        return False, error_msg, []


def get_base_data_last_updateDT(conn):
    """
    get the last update datetime for base data for the app

    args:
        conn: the connection to the database

    returns:
        Tuple of (success, error_message, last_updateDT)  
    """

    try:
        cursor = conn.cursor()
        
        # SQL query to fetch the last update datetime
        query = """
        SELECT MAX(LastUpdateDT) AS LastUpdateDT
        FROM [DM_MONTYNT\\dli2].PreprocessorBaseDataUpdate
        """
        
        # Execute query
        cursor.execute(query)
        
        # Process result
        row = cursor.fetchone()
        if row and row[0]:
            last_updateDT = row[0]
            return True, "", last_updateDT
        else:
            return False, "No update datetime found.", None
        
    except Exception as e:
        error_msg = f"Error retrieving base data last update datetime: {str(e)}"
        print(error_msg)
        return False, error_msg, None
    

def get_task_error_edits(conn, task_id):
    """
    Retrieve error edit records for a given task ID from PreprocessorErrorEdit table.
    
    Args:
        conn: Database connection
        task_id: ID of the task to retrieve error edits for
    Returns:
        Tuple of (success, error_message, results)
    """

    try:
        cursor = conn.cursor()
        
        # SQL query to fetch error edit records
        query = """
        SELECT *
        FROM [DM_MONTYNT\\dli2].PreprocessorErrorEdit
        WHERE TaskID = ?
        ORDER BY [FileRow], [DataSet] DESC, [Edit Field]
        """
        
        # Execute query with parameters
        cursor.execute(query, (task_id,))
        
        # Process results
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = []
        for row in rows:
            results.append(dict(zip(columns, row)))
        
        return True, "", results
        
    except Exception as e:
        error_msg = f"Error retrieving task error edits: {str(e)}"
        print(error_msg)
        return False, error_msg, []


def save_error_edit(conn, edit_data):
    """
    Save edits made to error records in the PreprocessorErrorEdit table.
    If the record already exists, update it; otherwise, insert a new record.
    If no changes need to be persisted, delete the record if it exists.

    Args:
        conn: Database connection
        edit_data: Dictionary containing edit information

    Returns:
        Tuple of (success, error_message)
    """
    try:
        cursor = conn.cursor()

        # Check if the new value is the same as the corresponding field's value
        if (
            (edit_data['edit_field'] == 'MfgPartNum' and edit_data['new_value'] == edit_data['mfg_part_num']) or
            (edit_data['edit_field'] == 'VendorPartNum' and edit_data['new_value'] == edit_data['vendor_part_num']) or
            (edit_data['edit_field'] == 'UOM' and edit_data['new_value'] == edit_data['uom']) or
            (edit_data['edit_field'] == 'QOE' and str(edit_data['new_value']) == str(edit_data['qoe']))
        ) and (
            edit_data['edit_field'] != 'DROP' and edit_data['is_drop'] == 0
        ):
            # If the new value is the same as the existing value, delete the record if it exists
            delete_query = """
            DELETE FROM [DM_MONTYNT\\dli2].PreprocessorErrorEdit
            WHERE TaskID = ? AND PKID_PrPRaw = ? AND [Edit Field] = ?
            """
            cursor.execute(delete_query, (
                edit_data['task_id'],
                edit_data['pkid'],
                edit_data['edit_field']
            ))
            conn.commit()
            rows_deleted = cursor.rowcount
            if rows_deleted > 0:
                return True, f"Record deleted as no changes needed to persist (rows deleted: {rows_deleted})"
            else:
                return True, "No changes to persist and no record to delete"

        # Use MERGE to handle insert or update
        merge_query = """
        MERGE INTO [DM_MONTYNT\\dli2].PreprocessorErrorEdit AS target
        USING (VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)) AS source (
            TaskID, DataSet, FileRow, [Mfg Part Num], [Vendor Part Num], [UOM], [QOE], 
            [Contract Number], PKID_PrPRaw, isDrop, isWrong, [Edit Field], [New Value]
        )
        ON target.TaskID = source.TaskID 
           AND target.PKID_PrPRaw = source.PKID_PrPRaw 
           AND target.[Edit Field] = source.[Edit Field]
        WHEN MATCHED THEN
            UPDATE SET 
                target.DataSet = source.DataSet,
                target.FileRow = source.FileRow,
                target.[Mfg Part Num] = source.[Mfg Part Num],
                target.[Vendor Part Num] = source.[Vendor Part Num],
                target.[UOM] = source.[UOM],
                target.[QOE] = source.[QOE],
                target.[Contract Number] = source.[Contract Number],
                target.isDrop = source.isDrop,
                target.isWrong = source.isWrong,
                target.[New Value] = source.[New Value],
                target.[UpdateDT] = GETDATE()
        WHEN NOT MATCHED THEN
            INSERT (
                TaskID, DataSet, FileRow, [Mfg Part Num], [Vendor Part Num], [UOM], [QOE], 
                [Contract Number], PKID_PrPRaw, isDrop, isWrong, [Edit Field], [New Value], 
                [CreateDT], [UpdateDT], [EditBy]
            )
            VALUES (
                source.TaskID, source.DataSet, source.FileRow, source.[Mfg Part Num], 
                source.[Vendor Part Num], source.[UOM], source.[QOE], source.[Contract Number], 
                source.PKID_PrPRaw, source.isDrop, source.isWrong, source.[Edit Field], 
                source.[New Value], GETDATE(), GETDATE(), ?
            );
        """

        # Execute the query with parameters
        cursor.execute(merge_query, (
            edit_data['task_id'],
            edit_data['data_set'],
            edit_data['file_row'],
            edit_data['mfg_part_num'],
            edit_data['vendor_part_num'],
            edit_data['uom'],
            edit_data['qoe'],
            edit_data['contract_number'],
            edit_data['pkid'],
            edit_data['is_drop'],
            edit_data['is_wrong'],
            edit_data['edit_field'],
            edit_data['new_value'],
            edit_data['edit_by']
        ))

        conn.commit()
        return True, ""

    except Exception as e:
        if conn:
            conn.rollback()
        error_msg = f"Error saving error edit: {str(e)}"
        print(error_msg)
        return False, error_msg
    

def revert_error_edit(conn, task_id, pkid, data_set, file_row, contract_number):
    """
    Revert all edits for a specific error record by deleting from PreprocessorErrorEdit table
    
    Args:
        conn: Database connection
        task_id: Task ID
        pkid: PKID_PrPRaw value
        data_set: Data set value
        file_row: File row number
        contract_number: Contract number
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        cursor = conn.cursor()
        
        # Delete all records from PreprocessorErrorEdit table for this row
        delete_query = """
        DELETE FROM [DM_MONTYNT\\dli2].PreprocessorErrorEdit
        WHERE TaskID = ? AND PKID_PrPRaw = ? AND DataSet = ? 
          AND FileRow = ? AND [Contract Number] = ?
        """
        
        cursor.execute(delete_query, (
            task_id, pkid, data_set, file_row, contract_number
        ))
        
        # Get the number of affected rows
        rows_affected = cursor.rowcount
        
        conn.commit()
        
        return True, f"Successfully reverted {rows_affected} edit(s)"
        
    except Exception as e:
        if conn:
            conn.rollback()
        error_msg = f"Error reverting error edits: {str(e)}"
        print(error_msg)
        return False, error_msg


def delete_all_error_edits(conn, task_id):
    """    Delete all error edits for a specific task ID from PreprocessorErrorEdit table.
    
    Args:
        conn: Database connection
        task_id: Task ID to delete edits for    
    Returns:
        Tuple of (success, error_message)
    """

    try:
        cursor = conn.cursor()
        # Delete all records from PreprocessorErrorEdit table for this task ID
        delete_query = """
        DELETE FROM [DM_MONTYNT\\dli2].PreprocessorErrorEdit
        WHERE TaskID = ?
        """
        cursor.execute(delete_query, (task_id,))
        conn.commit()
        return True, "All error edits deleted successfully"
    except Exception as e:
        if conn:
            conn.rollback()
        error_msg = f"Error deleting all error edits for task {task_id}: {str(e)}"
        print(error_msg)
        return False, error_msg


def check_task_owner(conn, task_id, user_id):
    """
    Check if the user is the owner of the task.
    Args:
        conn: Database connection
        task_id: ID of the task to check 
        user_id: ID of the user to check ownership
    Returns:
        Tuple of (success, error_message, is_owner)
    """
    try:
        cursor = conn.cursor()
        # Query to check if the user is the owner of the task
        query = """
        SELECT UserID FROM [DM_MONTYNT\\dli2].PreprocessorHeader
        WHERE TaskID = ? AND UserID = ?
        """
        
        cursor.execute(query, (task_id, user_id))
        result = cursor.fetchone()
        if result:
            is_owner = True
        else:
            is_owner = False
        return True, "", is_owner
    except Exception as e:
        error_msg = f"Error checking task ownership for task {task_id}: {str(e)}"
        print(error_msg)
        return False, error_msg, False
    

def get_sync_percentages(conn, task_id):
    """
    Get synchronization percentages for a specific task
    
    Args:
        conn: Database connection
        task_id: ID of the task to get sync percentages for
        
    Returns:
        Tuple of (success, error_message, percentages_dict)
    """
    try:
        cursor = conn.cursor()
        percentages = {
            'maxSync': 0.00,
            'ccxTpSync': 0.00,
            'ccxChangesSync': None,
            'inforTpSync': 0.00,
            'inforChangesSync': None
        }
        
        # Query 1: TP sync % for CCX
        query1 = """
        SELECT 
            TaskID,
            CAST(100.0 * SUM(CASE WHEN Synced = 'Synced' THEN 1 ELSE 0 END) / COUNT(1) AS DECIMAL(10,2)) AS Synced_pct,
            CAST(100.0 * SUM(CASE WHEN [file row action] = 'Execute' THEN 1 ELSE 0 END) / COUNT(1) AS DECIMAL(10,2)) AS Max_Synced_pct
        FROM [DM_MONTYNT\\dli2].vw_PreprocessorSyncInspectionTP
        WHERE TaskID = ?
        GROUP BY TaskID
        """
        cursor.execute(query1, (task_id,))
        row = cursor.fetchone()
        if row:
            percentages['ccxTpSync'] = row[1] if row[1] else 0.00
            max_sync = row[2] if row[2] else 0.00
            percentages['maxSync'] = max(percentages['maxSync'], max_sync)
        
        # Query 2: TP sync % for Infor
        query2 = """
        SELECT 
            TaskID,
            CAST(100.0 * SUM(CASE WHEN Synced = 'Synced' THEN 1 ELSE 0 END) / COUNT(1) AS DECIMAL(10,2)) AS Synced_pct,
            CAST(100.0 * SUM(CASE WHEN [file row action] = 'Execute' THEN 1 ELSE 0 END) / COUNT(1) AS DECIMAL(10,2)) AS Max_Synced_pct
        FROM [DM_MONTYNT\\dli2].vw_PreprocessorSyncInspectionTP2
        WHERE TaskID = ?
        GROUP BY TaskID
        """
        cursor.execute(query2, (task_id,))
        row = cursor.fetchone()
        if row:
            percentages['inforTpSync'] = row[1] if row[1] else 0.00
            max_sync = row[2] if row[2] else 0.00
            percentages['maxSync'] = max(percentages['maxSync'], max_sync)
        
        # Query 3: Export changes sync % for CCX
        query3 = """
        SELECT 
            TaskID,
            CAST(100.0 * SUM(CASE WHEN Synced = 'Synced' THEN 1 ELSE 0 END) / COUNT(*) AS DECIMAL(10,2)) AS Synced_Percentage
        FROM [DM_MONTYNT\\dli2].vw_PreprocessorSyncInspection
        WHERE TaskID = ?
        GROUP BY TaskID
        """
        cursor.execute(query3, (task_id,))
        row = cursor.fetchone()
        if row:
            percentages['ccxChangesSync'] = row[1] if row[1] else 0.00
        
        # Query 4: Export changes sync % for Infor
        query4 = """
        SELECT 
            TaskID,
            CAST(100.0 * SUM(CASE WHEN Synced = 'Synced' THEN 1 ELSE 0 END) / COUNT(*) AS DECIMAL(10,2)) AS Synced_Percentage
        FROM [DM_MONTYNT\\dli2].vw_PreprocessorSyncInspection2
        WHERE TaskID = ?
        GROUP BY TaskID
        """
        cursor.execute(query4, (task_id,))
        row = cursor.fetchone()
        if row:
            percentages['inforChangesSync'] = row[1] if row[1] else 0.00
            
        return True, "", percentages
        
    except Exception as e:
        error_msg = f"Error getting sync percentages: {str(e)}"
        current_app.logger.error(error_msg) if 'current_app' in globals() else print(error_msg)
        return False, error_msg, None


def get_inspection_summary_count(conn, task_id):
    """
    Get inspection summary counts for a specific task from the sync inspection state view
    
    Args:
        conn: Database connection
        task_id: ID of the task to get inspection summary for
        
    Returns:
        Tuple of (success, error_message, summary_dict)
    """
    try:
        cursor = conn.cursor()
        
        query = """
        SELECT 
            TaskID,
            Total_TP_Lines,
            Total_Executable_TP_Lines,
            Total_Error_TP_Lines,
            Total_Exported_Change_Lines,
            Total_Executable_Exported_Change_Lines,
            Total_Error_Exported_Change_Lines,
            Total_Error_Count,
            Total_TP_Error_Count,
            Total_CCX_Error_Count
        FROM [DM_MONTYNT\\dli2].vw_PreprocessorSyncInspectionStat
        WHERE TaskID = ?
        ORDER BY Total_TP_Lines desc, Total_Exported_Change_Lines desc
        """
        
        cursor.execute(query, (task_id,))
        row = cursor.fetchone()
        
        if not row:
            return False, f"No inspection data found for TaskID: {task_id}", None
        
        # Convert to dictionary with proper null handling
        summary = {
            'taskId': row[0],
            'totalTPLines': row[1] if row[1] is not None else 'N/A',
            'totalExecutableTPLines': row[2] if row[2] is not None else 'N/A',
            'totalErrorTPLines': row[3] if row[3] is not None else 'N/A',
            'totalExportedChangeLines': row[4] if row[4] is not None else 'N/A',
            'totalExecutableExportedChangeLines': row[5] if row[5] is not None else 'N/A',
            'totalErrorExportedChangeLines': row[6] if row[6] is not None else 'N/A',
            'totalErrorCount': row[7] if row[7] is not None else 'N/A',
            'totalTPErrorCount': row[8] if row[8] is not None else 'N/A',
            'totalCCXErrorCount': row[9] if row[9] is not None else 'N/A'
        }
        
        return True, None, summary
        
    except Exception as e:
        error_msg = f"Error getting inspection summary: {str(e)}"
        current_app.logger.error(error_msg) if 'current_app' in globals() else print(error_msg)
        return False, error_msg, None


def get_affected_contract_by_task(conn, task_id):
    """
    Get affected contracts by task ID with operations count
    
    Args:
        conn: Database connection
        task_id: ID of the task to get affected contracts for
        
    Returns:
        Tuple of (success, error_message, contracts_list)
    """
    try:
        cursor = conn.cursor()
        
        query = """
        SELECT 
            taskID, 
            [Contract Number], 
            count(1) as Total_committed_operations,
            sum(case when [File Row Action] = 'Execute' then 1 else 0 End) as Executable_operations, 
            sum(case when [File Row Action] = 'Pending' then 1 else 0 End) as Non_executable_operations
        FROM [DM_MONTYNT\\dli2].PreprocessorCommitLine
        WHERE [Actual Action] not in ('Create then Expire (Expire)', 'Update (Existing)')
            AND taskID = ?
        GROUP BY taskID, [Contract Number]
        ORDER BY [Contract Number]
        """
        
        cursor.execute(query, (task_id,))
        rows = cursor.fetchall()
        
        contracts = []
        for row in rows:
            contract = {
                'taskId': row[0],
                'contractNumber': row[1] if row[1] is not None else 'N/A',
                'totalCommittedOperations': row[2] if row[2] is not None else 0,
                'executableOperations': row[3] if row[3] is not None else 0,
                'nonExecutableOperations': row[4] if row[4] is not None else 0
            }
            contracts.append(contract)
        
        return True, None, contracts
        
    except Exception as e:
        error_msg = f"Error getting affected contracts: {str(e)}"
        current_app.logger.error(error_msg) if 'current_app' in globals() else print(error_msg)
        return False, error_msg, None
    

def get_tp_row_sync_by_taskid(conn, task_id):
    """Return TP row sync details (TP vs CCX and TP vs Infor comparisons) for a task.

    Args:
        conn: Active DB connection
        task_id: Task identifier (string/int)

    Returns:
        Tuple[bool, str|None, list|None]: (success, error_message, rows)
    """
    try:
        cursor = conn.cursor()
        query = """
        SELECT 
            ccx_tp.TaskID, 
            ccx_tp.PKID, 
            ccx_tp.[Intended Action], 
            ccx_tp.[File Row Action],
            ccx_tp.[Actual Action], 
            ccx_tp.[ERP Vendor ID], 
            ccx_tp.[Contract Number],
            ccx_tp.[Mfg Part Num], 
            ccx_tp.[UOM], 
            UOM_INFOR,
            Item as Item_TP,
            Item_Infor,
            ccx_tp.Item_Matching_Flag AS Item_Matching_Flag_CCX, 
            infor_tp.Item_Matching_Flag AS Item_Matching_Flag_Infor,
            ccx_tp.Synced AS TP_CCX_Synced, 
            infor_tp.Synced AS TP_Infor_Synced,
            ccx_tp.QOE AS QOE_TP, 
            QOE_CCX, 
            QOE_Infor,
            ccx_tp.[Contract Price] AS Price_TP, 
            PRICE AS Price_CCX, 
            BaseCost AS Price_Infor,
            ccx_tp.[Vendor Part Num] AS VendorPartNum_TP, 
            VENDOR_PART_NUMBER AS VendorPartNum_CCX, 
            VendorItem AS VendorPartNum_Infor,
            ccx_tp.[Effective Date] AS EffectiveDate_TP, 
            ITEM_PRICE_START_DATE AS EffectiveDate_CCX, 
            EffectiveDate AS EffectiveDate_Infor,
            ccx_tp.[Expiration Date] AS ExpirationDate_TP, 
            ITEM_PRICE_END_DATE AS ExpirationDate_CCX, 
            ExpirationDate AS ExpirationDate_Infor,
            ccx_tp.[Description] AS Description_TP, 
            PART_DESCRIPTION AS Description_CCX, 
            ItemDescription AS Description_Infor,
            CCX_LAST_REFRESH, 
            infor_record_update,

            CASE WHEN ccx_tp.QOE = QOE_CCX THEN 1 ELSE 0 END AS Match_QOE_TP_CCX,
            CASE WHEN ccx_tp.[Contract Price] = PRICE THEN 1 ELSE 0 END AS Match_Price_TP_CCX,
            CASE WHEN ccx_tp.[Vendor Part Num] = VENDOR_PART_NUMBER THEN 1 ELSE 0 END AS Match_VendorPartNum_TP_CCX,
            CASE WHEN ccx_tp.[Effective Date] <= ITEM_PRICE_START_DATE THEN 1 ELSE 0 END AS Match_EffectiveDate_TP_CCX,
            CASE WHEN ccx_tp.[Expiration Date] = ITEM_PRICE_END_DATE THEN 1 ELSE 0 END AS Match_ExpirationDate_TP_CCX,
            CASE WHEN ccx_tp.[Description] = PART_DESCRIPTION THEN 1 ELSE 0 END AS Match_Description_TP_CCX,

            CASE WHEN ccx_tp.QOE = QOE_Infor THEN 1 ELSE 0 END AS Match_QOE_TP_Infor,
            CASE WHEN ccx_tp.[Contract Price] = BaseCost THEN 1 ELSE 0 END AS Match_Price_TP_Infor,
            CASE WHEN ccx_tp.[Vendor Part Num] COLLATE Latin1_General_CS_AS = VendorItem COLLATE Latin1_General_CS_AS THEN 1 ELSE 0 END AS Match_VendorPartNum_TP_Infor,
            -1 AS Match_EffectiveDate_TP_Infor,
            CASE WHEN ccx_tp.[Expiration Date] = ExpirationDate THEN 1 ELSE 0 END AS Match_ExpirationDate_TP_Infor,
            -1 AS Match_Description_TP_Infor

        FROM [DM_MONTYNT\\dli2].vw_PreprocessorSyncInspectionTP AS ccx_tp
        JOIN [DM_MONTYNT\\dli2].vw_PreprocessorSyncInspectionTP2 AS infor_tp
            ON ccx_tp.TaskID = infor_tp.TaskID
           AND ccx_tp.PKID = infor_tp.PKID
        WHERE ccx_tp.TaskID = ?
        ORDER BY TP_CCX_Synced, TP_Infor_Synced, ccx_tp.[File Row Action] desc, Item_Matching_Flag_CCX desc, Item_Matching_Flag_Infor desc,
            ccx_tp.PKID
        """
        cursor.execute(query, (task_id,))
        rows = cursor.fetchall()
        columns = [c[0] for c in cursor.description]
        out = []
        for row in rows:
            cleaned = [fix_encoding(v) for v in row]
            out.append(dict(zip(columns, cleaned)))
        return True, None, out
    except Exception as e:
        msg = f"Error getting TP rows sync for {task_id}: {str(e)}"
        try:
            current_app.logger.error(msg)
        except Exception:
            print(msg)
        return False, msg, None


def get_exported_row_sync_by_taskid(conn, task_id):
    """
    Return Exported Rows sync details for a task.

    Uses the query provided, substituting:
      - [Exported Date] (from try_convert(date, ExportedDT))
      - [Final Action] (IIF([Final Rank]=1,'Execute','Masked'))

    Args:
        conn: Active DB connection
        task_id: Task identifier (string/int)

    Returns:
        Tuple[bool, str|None, list|None]: (success, error_message, rows)
    """
    try:
        cursor = conn.cursor()
        query = """
        SELECT 
            ccx_tp.TaskID, 
            ccx_tp.PKID, 
            try_convert(date, ccx_tp.[ExportedDT]) as [Exported Date], --replace [Intended Action]
            IIF(ccx_tp.[Final Rank] = 1, 'Execute', 'Masked') as [Final Action], -- replace [File Row Action]
            ccx_tp.[Actual Action], 
            ccx_tp.[ERP Vendor ID (CCX Sync)] AS [ERP Vendor ID], 
            ccx_tp.[Contract Number],
            ccx_tp.[Mfg Part Num], 
            ccx_tp.[UOM], 
            UOM_INFOR,
            Item, Item_Infor,
            ccx_tp.Item_Matching_Flag AS Item_Matching_Flag_CCX, 
            infor_tp.Item_Matching_Flag AS Item_Matching_Flag_Infor,
            ccx_tp.Synced AS TP_CCX_Synced, 
            infor_tp.Synced AS TP_Infor_Synced,
            ccx_tp.QOE AS QOE_TP, 
            QOE_CCX, 
            QOE_Infor,
            ccx_tp.[Contract Price] AS Price_TP, 
            PRICE AS Price_CCX, 
            BaseCost AS Price_Infor,
            ccx_tp.[Vendor Part Num] AS VendorPartNum_TP, 
            VENDOR_PART_NUMBER AS VendorPartNum_CCX, 
            VendorItem AS VendorPartNum_Infor,
            ccx_tp.[Effective Date] AS EffectiveDate_TP, 
            ITEM_PRICE_START_DATE AS EffectiveDate_CCX, 
            EffectiveDate AS EffectiveDate_Infor,
            ccx_tp.[Expiration Date] AS ExpirationDate_TP, 
            ITEM_PRICE_END_DATE AS ExpirationDate_CCX, 
            ExpirationDate AS ExpirationDate_Infor,
            ccx_tp.[Description] AS Description_TP, 
            PART_DESCRIPTION AS Description_CCX, 
            ItemDescription AS Description_Infor,
            CCX_LAST_REFRESH, 
            infor_record_update,

            -- Comparisons TP vs CCX
            CASE WHEN ccx_tp.QOE = QOE_CCX THEN 1 ELSE 0 END AS Match_QOE_TP_CCX,
            CASE WHEN ccx_tp.[Contract Price] = PRICE THEN 1 ELSE 0 END AS Match_Price_TP_CCX,
            CASE WHEN ccx_tp.[Vendor Part Num] = VENDOR_PART_NUMBER THEN 1 ELSE 0 END AS Match_VendorPartNum_TP_CCX,
            CASE WHEN ccx_tp.[Effective Date] <= ITEM_PRICE_START_DATE THEN 1 ELSE 0 END AS Match_EffectiveDate_TP_CCX,
            CASE WHEN ccx_tp.[Expiration Date] = ITEM_PRICE_END_DATE THEN 1 ELSE 0 END AS Match_ExpirationDate_TP_CCX,
            CASE WHEN ccx_tp.[Description] = PART_DESCRIPTION THEN 1 ELSE 0 END AS Match_Description_TP_CCX,

            -- Comparisons TP vs Infor
            CASE WHEN ccx_tp.QOE = QOE_Infor THEN 1 ELSE 0 END AS Match_QOE_TP_Infor,
            CASE WHEN ccx_tp.[Contract Price] = BaseCost THEN 1 ELSE 0 END AS Match_Price_TP_Infor,
            CASE WHEN ccx_tp.[Vendor Part Num] COLLATE Latin1_General_CS_AS = VendorItem COLLATE Latin1_General_CS_AS THEN 1 ELSE 0 END AS Match_VendorPartNum_TP_Infor,
            -1 AS Match_EffectiveDate_TP_Infor,
            CASE WHEN ccx_tp.[Expiration Date] = ExpirationDate THEN 1 ELSE 0 END AS Match_ExpirationDate_TP_Infor,
            -1 Match_Description_TP_Infor

        FROM [DM_MONTYNT\\dli2].vw_PreprocessorSyncInspection AS ccx_tp
        JOIN [DM_MONTYNT\\dli2].vw_PreprocessorSyncInspection2 AS infor_tp
            ON ccx_tp.TaskID = infor_tp.TaskID
           AND ccx_tp.PKID = infor_tp.PKID
        WHERE ccx_tp.TaskID = ?
        ORDER BY TP_CCX_Synced, TP_Infor_Synced, Item_Matching_Flag_CCX desc, Item_Matching_Flag_Infor desc,
        ccx_tp.PKID
        """
        cursor.execute(query, (task_id,))
        rows = cursor.fetchall()
        columns = [c[0] for c in cursor.description]
        out = []
        for row in rows:
            cleaned = [fix_encoding(v) for v in row]
            out.append(dict(zip(columns, cleaned)))
        return True, None, out
    except Exception as e:
        msg = f"Error getting exported rows sync for {task_id}: {str(e)}"
        try:
            current_app.logger.error(msg)
        except Exception:
            print(msg)
        return False, msg, None
    

def mark_completed_by_task_id(conn, task_id, user_id):
    """ Mark a task as completed in the database.
    update table preprocessorHeader on Status2 = 'Completed' and UpdateDT = GETDATE() and completedBy = user_id
    Args:
        conn: Active DB connection
        task_id: Task identifier (string/int)
        user_id: ID of the user performing the action
    Returns:
        Tuple[bool, str|None]: (success, error_message)
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE [DM_MONTYNT\\dli2].PreprocessorHeader
            SET Status = 'Completed', Status2 = 'Completed', UpdateDT = GETDATE(), CompletedBy = ?
            WHERE TaskID = ? AND Status2 != 'Completed' AND Status == 'Exported'
        """, (user_id, task_id))
        if cursor.rowcount == 0:
            return False, "Task not found or already completed."
        conn.commit()
        return True, ""
    except Exception as e:
        conn.rollback()
        error_msg = f"Error marking task {task_id} as completed: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg


def add_completion_comment_by_task_id(conn, task_id, comment):
    """ Add a completion comment to a task in the database.
    insert into table PreprocessorCompletionComment with TaskID = task_id, [Completion Comment] = comment, CreateDT = GETDATE()
    Args:
        conn: Active DB connection
        task_id: Task identifier (string/int)
        comment: Comment to be added
    Returns:
        Tuple[bool, str|None]: (success, error_message)
    """
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO [DM_MONTYNT\\dli2].PreprocessorCompletionComment (TaskID, [Completion Comment], CreateDT)
            VALUES (?, ?, GETDATE())
        """, (task_id, comment))
        conn.commit()
        return True, ""
    except Exception as e:
        conn.rollback()
        error_msg = f"Error adding completion comment for task {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg


def get_completion_timestamps_by_task_id(conn, task_id):
    """ Retrieve timestamps for task creation, export, and completed on for a specific task ID from the database.
        createDT and completedDT (updateDT) are from PreprocessorHeader table for the task
        exportedDT is the maximum exportedDT from PreprocessorExported table for the task

    Args:
        conn: Active DB connection
        task_id: Task identifier (string/int)
    
    Returns:
        success, error_msg, (createDT, exportDT, completedDT)
    """
    try:
        cursor = conn.cursor()
        query = """
        SELECT createDT,
               (SELECT MAX(ExportedDT) FROM [DM_MONTYNT\\dli2].PreprocessorExported WHERE TaskID = ?) as exportedDT,
               updateDT as completedDT      
        FROM [DM_MONTYNT\\dli2].PreprocessorHeader
        WHERE TaskID = ?
        """
        cursor.execute(query, (task_id, task_id))
        row = cursor.fetchone()
        if not row:
            return False, "Task not found.", None
        
        timestamps = {
            'createDT': row[0],
            'exportedDT': row[1],
            'completedDT': row[2]
        }
        return True, "", timestamps
    
    except Exception as e:
        error_msg = f"Error retrieving timestamps for task {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None


def get_completion_count_by_task_id(conn, task_id):
    """ Retrieve counts for total TP items, and total item changed for a specific task ID
    we will get total TP items from PreprocessorProcessedRaw where TaskID = task_id
    we will get total exported changes from PreprocessorCommitLine where TaskID = task_id and Exported = 'Yes'

    Args:
        conn: Active DB connection
        task_id: Task identifier (string/int)
    
    Returns:
        success, error_msg, (totalTPItems, totalAffectedItems)
    """
    try:
        cursor = conn.cursor()
        query = """
        SELECT 
            (SELECT COUNT(1) FROM [DM_MONTYNT\\dli2].PreprocessorProcessedRaw WHERE TaskID = ? AND Dataset = 'TP') as totalProcessedItems,
            (SELECT COUNT(1) FROM (
                SELECT DISTINCT FileRow, [Contract Number]
                FROM [DM_MONTYNT\\dli2].PreprocessorCommitLine WHERE TaskID = ? AND Exported = 'Yes') [x]) as totalAffectedItems
        """
        cursor.execute(query, (task_id, task_id))
        row = cursor.fetchone()
        if not row:
            return False, "Task not found.", None
        
        counts = {
            'totalTPItems': row[0],
            'totalAffectedItems': row[1]
        }
        return True, "", counts
    
    except Exception as e:
        error_msg = f"Error retrieving counts for task {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None


def get_completion_contract_count_by_task_id(conn, task_id):
    """Retrieve count of unique contracts affected for a specific task ID from the database.
    
    Args:
        conn: Active DB connection
        task_id: Task identifier (string/int)
    Returns:
        success, error_msg, totalAffectedContracts
    """

    try:
        cursor = conn.cursor()
        query = """
        SELECT COUNT(DISTINCT [Contract Number]) as totalAffectedContracts
        FROM [DM_MONTYNT\\dli2].PreprocessorCommitLine
        WHERE TaskID = ? AND Exported = 'Yes'
        """
        cursor.execute(query, (task_id,))
        row = cursor.fetchone()
        if not row:
            return False, "Task not found.", None
        
        total_affected_contracts = row[0] if row[0] is not None else 0
        return True, "", total_affected_contracts
        
    except Exception as e:
        error_msg = f"Error retrieving affected contracts count for task {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None


def get_completion_comments_by_task_id(conn, task_id):
    """Retrieve completion comments for a specific task ID from the database.
    
    Args:
        conn: Active DB connection
        task_id: Task identifier (string/int)
    
    Returns:
        success, error_msg, [comments_list]
    """
    try:
        cursor = conn.cursor()
        query = """
        SELECT [Completion Comment], CreateDT
        FROM [DM_MONTYNT\\dli2].PreprocessorCompletionComment
        WHERE TaskID = ?
        ORDER BY CreateDT DESC
        """
        cursor.execute(query, (task_id,))
        rows = cursor.fetchall()
        
        comments = []
        for row in rows:
            comments.append({
                'comment': row[0],
                'createDT': row[1]
            })
        
        return True, "", comments
        
    except Exception as e:
        error_msg = f"Error retrieving completion comments for task {task_id}: {str(e)}"
        current_app.logger.error(error_msg)
        return False, error_msg, None


def log_task_file(conn, task_id, file_type, file_path):
    """
    Insert a record into PreprocessorTaskFile to track generated files.

    Args:
        conn: DB connection
        task_id: str
        file_type: str (e.g., 'TP_REPROCESS')
        file_path: str (absolute or relative path)
    Returns:
        (success: bool, msg: str)
    """
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO [DM_MONTYNT\\dli2].PreprocessorTaskFile (TaskID, FileType, FilePath)
            VALUES (?, ?, ?)
        """, (task_id, file_type, file_path))
        conn.commit()
        return True, ""
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        msg = f"Error logging task file: {str(e)}"
        try:
            current_app.logger.error(msg)
        except Exception:
            print(msg)
        return False, msg


def get_latest_task_file(conn, task_id, file_type='TP_REPROCESS'):
    """Get the most recent file path for a given task and file type.

    Args:
        conn: Active DB connection
        task_id: Task identifier (string/int)
        file_type: File type to filter by, default 'TP_REPROCESS'

    Returns:
        (success: bool, msg: str, file_path: Optional[str])
    """
    try:
        cursor = conn.cursor()
        query = (
            """
            SELECT TOP 1 FilePath
            FROM [DM_MONTYNT\\dli2].PreprocessorTaskFile
            WHERE TaskID = ? AND FileType = ?
            ORDER BY CreateDT DESC
            """
        )
        cursor.execute(query, (task_id, file_type))
        row = cursor.fetchone()
        if not row:
            return True, "No file found", None
        return True, "", row[0]
    except Exception as e:
        msg = f"Error retrieving latest task file for {task_id}: {str(e)}"
        try:
            current_app.logger.error(msg)
        except Exception:
            pass
        return False, msg, None