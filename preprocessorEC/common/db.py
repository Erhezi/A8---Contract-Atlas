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
        
        return True, table_name
    
    except Exception as e:
        print(f"Database error: {str(e)}")
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
                    VENDOR_PART_NUMBER,
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


def match_to_infor_contract_lines(temp_table, conn):
    """
    Match uploaded items to Infor Contract Lines
    
    Args:
        temp_table: Name of the temporary table containing upload data
        conn: Database connection (optional, will create one if None)
        
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
                        AND ContractLineState = 'Active'
                        AND OnHold = 'No'
                        AND ActiveLine = 'Yes'
                        AND ExpirationDate >= getdate()
                ) as infor
            INNER JOIN 
                {temp_table} as temp
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
            results.append(dict(zip(columns, row)))
        
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
        conn: Database connection (optional, will create one if None)
        
    Returns:
        Tuple of (success, error_message, results)
    """
    try:
        cursor = conn.cursor()
        
        # Execute the matching query
        match_query = f"""
            SELECT 
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
            results.append(dict(zip(columns, row)))
        
        # Group by item number (Item Master side)
        item_summary = {}
        for item in results:
            item_num = item['item_number_infor']
            if item_num not in item_summary:
                item_summary[item_num] = {
                    'item_number': item_num,
                    'description': item['description_infor'],
                    'manufacturer_name': item['manufacturer_name_infor'],
                    'total_matches': 0,
                    'exact_matches': 0,
                    'items': []
                }
            
            # Count this item
            item_summary[item_num]['total_matches'] += 1
            if item['same_mfg_part_num'] == 1:
                item_summary[item_num]['exact_matches'] += 1
            
            # Add this item to the items list
            item_summary[item_num]['items'].append(item)
        
        # Convert to a list
        item_list = list(item_summary.values())
        
        return True, "", item_list
        
    except Exception as e:
        error_msg = f"Error in match_to_item_master: {str(e)}"
        current_app.logger.error(error_msg)
        if conn and 'conn' in locals() and not conn.closed:
            conn.close()
        return False, error_msg, None