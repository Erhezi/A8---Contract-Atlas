import os
import uuid
from datetime import datetime, date

import pandas as pd
from flask import (Blueprint, request, jsonify, current_app,
                   render_template, send_file)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename

from ..common.db import get_db_connection, get_contract_summary
from ..common.session import (
    get_prepare_tp_files,
    store_prepare_tp_files,
    clear_prepare_tp_final_file,
    store_prepare_tp_final_file
)
from ..common.utils import load_organizations_map


prepare_tp_bp = Blueprint(
    'prepare_tp',
    __name__,
    url_prefix='/prepare-tp',
    template_folder='templates',
)


REQUIRED_COLUMNS = [
    'Mfg Part Num',
    'Vendor Part Num',
    'Buyer Part Num',
    'Description',
    'Contract Price',
    'UOM',
    'QOE',
    'Effective Date',
    'Expiration Date',
]

INTENDED_ACTIONS = [
    'Add/Update Item to Existing Contract',
    'New Contract',
    'Expire Item from Existing Contract',
]

SOURCE_TYPES = ['GPO', 'Local']

ALLOWED_EXTENSIONS = {'xlsx'}

OUTPUT_COLUMNS = [
    'Mfg Part Num',
    'Vendor Part Num',
    'Buyer Part Num',
    'Description',
    'Contract Price',
    'UOM',
    'QOE',
    'Effective Date',
    'Expiration Date',
    'Organization',
    'Contract Number',
    'ERP Vendor ID',
    'Source Contract Type',
    'Intended Action',
]

INTENDED_ACTION_TRANSLATIONS = {
    'Add/Update Item to Existing Contract': 'Upsert',
    'New Contract': 'Upsert',
    'Expire Item from Existing Contract': 'Expire',
    'Upsert': 'Upsert',
    'Expire': 'Expire',
}


def _translate_intended_action(value):
    if value is None:
        return None
    return INTENDED_ACTION_TRANSLATIONS.get(value, value)


def _format_output_dataframe(df):
    df = df.copy()

    if 'Contract Price' not in df.columns and 'Price' in df.columns:
        df = df.rename(columns={'Price': 'Contract Price'})

    if 'Vendor ERP ID' in df.columns and 'ERP Vendor ID' not in df.columns:
        df = df.rename(columns={'Vendor ERP ID': 'ERP Vendor ID'})

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = None

    ordered_columns = OUTPUT_COLUMNS
    return df[ordered_columns]


def _user_storage_dir(user_id):
    base_dir = os.path.join(current_app.root_path, 'temp_files', 'prepare_tp', str(user_id))
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _labeled_dir(user_id):
    labeled_dir = os.path.join(_user_storage_dir(user_id), 'labeled')
    os.makedirs(labeled_dir, exist_ok=True)
    return labeled_dir


def _allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _safe_remove(path):
    if not path:
        return
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            current_app.logger.warning('Failed to remove file at %s', path)


def _read_dataframe(file_path, file_ext):
    if file_ext == 'csv':
        df = pd.read_csv(file_path, dtype=str)
    else:
        df = pd.read_excel(file_path, dtype=str)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df


def _serialize_file_record(record):
    serialized = {
        'id': record['id'],
        'filename': record['filename'],
        'status': record['status'],
        'row_count': record.get('row_count', 0),
        'metadata': record.get('metadata', {}),
        'current_contract_end_date': record.get('current_contract_end_date'),
        'committed_at': record.get('committed_at'),
    }
    return serialized


def _get_file_records(user_id):
    return list(get_prepare_tp_files(user_id))


def _store_file_records(user_id, records):
    store_prepare_tp_files(user_id, records)


@prepare_tp_bp.route('/', methods=['GET'])
@login_required
def prepare_tp_home():
    organizations = load_organizations_map()
    return render_template(
        'prepare_tp.html',
        intended_actions=INTENDED_ACTIONS,
        source_types=SOURCE_TYPES,
        organizations=organizations,
    )


@prepare_tp_bp.route('/state', methods=['GET'])
@login_required
def get_state():
    user_id = current_user.id
    records = _get_file_records(user_id)
    payload = [_serialize_file_record(record) for record in records]
    return jsonify({
        'success': True,
        'files': payload,
        'intended_actions': INTENDED_ACTIONS,
        'source_types': SOURCE_TYPES,
        'organizations': load_organizations_map(),
    })


@prepare_tp_bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    user_id = current_user.id
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part provided.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected.'}), 400

    if not _allowed_file(file.filename):
        return jsonify({'success': False, 'message': 'Unsupported file type.'}), 400

    secure_name = secure_filename(file.filename)
    file_ext = secure_name.rsplit('.', 1)[1].lower()
    file_id = uuid.uuid4().hex
    storage_dir = _user_storage_dir(user_id)
    stored_filename = f"{file_id}_{secure_name}"
    file_path = os.path.join(storage_dir, stored_filename)

    file.save(file_path)

    try:
        df = _read_dataframe(file_path, file_ext)
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'success': False, 'message': str(exc)}), 400

    records = _get_file_records(user_id)
    record = {
        'id': file_id,
        'filename': secure_name,
        'stored_filename': stored_filename,
        'path': file_path,
        'status': 'uploaded',
        'row_count': len(df.index),
        'metadata': {
            'organization': None,
            'intended_action': None,
            'contract_number': None,
            'vendor_erp_id': None,
            'source_contract_type': None,
        },
        'current_contract_end_date': None,
        'committed_path': None,
        'committed_at': None,
    }
    records.append(record)
    _store_file_records(user_id, records)
    clear_prepare_tp_final_file(user_id)

    return jsonify({
        'success': True,
        'file': _serialize_file_record(record),
        'message': 'File uploaded successfully.'
    })


@prepare_tp_bp.route('/files/<file_id>/metadata', methods=['POST'])
@login_required
def update_metadata(file_id):
    user_id = current_user.id
    payload = request.get_json(silent=True) or {}

    records = _get_file_records(user_id)
    record = next((item for item in records if item['id'] == file_id), None)
    if not record:
        return jsonify({'success': False, 'message': 'File not found.'}), 404

    metadata = record.setdefault('metadata', {})

    if 'intended_action' in payload:
        intended_action = payload['intended_action']
        if intended_action and intended_action not in INTENDED_ACTIONS:
            return jsonify({'success': False, 'message': 'Invalid intended action.'}), 400
        metadata['intended_action'] = intended_action or None

    if 'contract_number' in payload:
        contract_number = payload['contract_number']
        metadata['contract_number'] = contract_number.strip() if contract_number else None

    if 'vendor_erp_id' in payload:
        vendor_erp_id = payload['vendor_erp_id']
        metadata['vendor_erp_id'] = vendor_erp_id.strip() if vendor_erp_id else None

    if 'source_contract_type' in payload:
        source_contract_type = payload['source_contract_type']
        if source_contract_type and source_contract_type not in SOURCE_TYPES:
            return jsonify({'success': False, 'message': 'Invalid source contract type.'}), 400
        metadata['source_contract_type'] = source_contract_type or None

    if 'organization' in payload:
        organization = payload['organization']
        if isinstance(organization, list):
            cleaned = [str(item).strip() for item in organization if str(item).strip()]
            metadata['organization'] = cleaned or None
        elif isinstance(organization, str):
            cleaned = organization.strip()
            metadata['organization'] = [cleaned] if cleaned else None
        else:
            metadata['organization'] = None

    if 'current_contract_end_date' in payload:
        record['current_contract_end_date'] = payload['current_contract_end_date'] or None

    _store_file_records(user_id, records)
    clear_prepare_tp_final_file(user_id)

    return jsonify({
        'success': True,
        'file': _serialize_file_record(record),
        'message': 'Metadata updated.'
    })


@prepare_tp_bp.route('/fetch-contract', methods=['GET'])
@login_required
def fetch_contract():
    contract_number = (request.args.get('contract_number') or '').strip()
    if not contract_number:
        return jsonify({'success': False, 'message': 'Contract number is required.'}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({'success': False, 'message': 'Unable to connect to database.'}), 500

    try:
        summary = get_contract_summary(conn, contract_number)
    finally:
        conn.close()

    if not summary:
        return jsonify({'success': False, 'message': 'Contract not found.'}), 404

    end_date = summary.get('CONTRACT_END_DATE')
    if isinstance(end_date, (datetime, date)):
        end_date_str = end_date.strftime('%Y-%m-%d')
    elif end_date is None:
        end_date_str = None
    else:
        end_date_str = str(end_date)

    data = {
        'contract_number': summary.get('CONTRACT_NUMBER') or contract_number,
        'source_contract_type': summary.get('SOURCE_CONTRACT_TYPE'),
        'vendor_erp_id': summary.get('VENDOR_ERP_NUMBER'),
        'current_contract_end_date': end_date_str,
    }

    return jsonify({'success': True, 'data': data})


def _ensure_commit_ready(record):
    metadata = record.get('metadata', {})
    missing = [
        key for key in ['intended_action', 'contract_number', 'vendor_erp_id', 'source_contract_type']
        if not metadata.get(key)
    ]
    if missing:
        raise ValueError(f"Missing metadata fields: {', '.join(missing)}")


@prepare_tp_bp.route('/files/<file_id>/commit', methods=['POST'])
@login_required
def commit_file(file_id):
    user_id = current_user.id
    records = _get_file_records(user_id)
    record = next((item for item in records if item['id'] == file_id), None)
    if not record:
        return jsonify({'success': False, 'message': 'File not found.'}), 404

    try:
        _ensure_commit_ready(record)
    except ValueError as exc:  # noqa: BLE001
        return jsonify({'success': False, 'message': str(exc)}), 400

    file_path = record['path']
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': 'Stored file is missing. Please re-upload.'}), 410

    file_ext = file_path.rsplit('.', 1)[1].lower()

    try:
        df = _read_dataframe(file_path, file_ext)
    except Exception as exc:  # noqa: BLE001
        return jsonify({'success': False, 'message': str(exc)}), 400

    df = df.copy()

    metadata = record['metadata']
    if metadata.get('organization'):
        org_value = metadata['organization']
        if isinstance(org_value, list):
            df['Organization'] = ', '.join(org_value)
        else:
            df['Organization'] = org_value
    df['Contract Number'] = metadata['contract_number']
    df['ERP Vendor ID'] = metadata['vendor_erp_id']
    df['Source Contract Type'] = metadata['source_contract_type']
    df['Intended Action'] = _translate_intended_action(metadata['intended_action'])
    if record.get('current_contract_end_date'):
        df['Current Contract End Date'] = record['current_contract_end_date']

    df = _format_output_dataframe(df)

    labeled_dir = _labeled_dir(user_id)
    output_filename = f"file_label_{file_id}.xlsx"
    output_path = os.path.join(labeled_dir, output_filename)
    df.to_excel(output_path, index=False)

    record['status'] = 'committed'
    record['committed_path'] = output_path
    record['committed_at'] = datetime.utcnow().isoformat()

    _store_file_records(user_id, records)
    clear_prepare_tp_final_file(user_id)

    return jsonify({
        'success': True,
        'file': _serialize_file_record(record),
        'message': 'File committed successfully.'
    })


@prepare_tp_bp.route('/files/<file_id>', methods=['DELETE'])
@login_required
def delete_file(file_id):
    user_id = current_user.id
    records = _get_file_records(user_id)
    index = next((idx for idx, item in enumerate(records) if item['id'] == file_id), None)
    if index is None:
        return jsonify({'success': False, 'message': 'File not found.'}), 404

    record = records.pop(index)

    for path_key in ['path', 'committed_path']:
        _safe_remove(record.get(path_key))

    _store_file_records(user_id, records)
    clear_prepare_tp_final_file(user_id)

    payload = [_serialize_file_record(item) for item in records]
    filename = record.get('filename')
    message = f"Removed {filename}." if filename else 'File removed.'

    return jsonify({'success': True, 'message': message, 'files': payload})


@prepare_tp_bp.route('/prepare-output', methods=['POST'])
@login_required
def prepare_output():
    user_id = current_user.id
    records = _get_file_records(user_id)
    committed = [record for record in records if record.get('status') == 'committed']

    if not committed:
        return jsonify({'success': False, 'message': 'No committed files available.'}), 400

    dataframes = []
    for record in committed:
        path = record.get('committed_path')
        if not path or not os.path.exists(path):
            return jsonify({'success': False, 'message': f"Prepared file missing for {record['filename']}."}), 500
        dataframes.append(pd.read_excel(path, dtype=str))

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = _format_output_dataframe(combined_df)
    if 'Intended Action' in combined_df.columns:
        combined_df['Intended Action'] = combined_df['Intended Action'].apply(_translate_intended_action)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    output_filename = f"prepared_tp_{timestamp}.xlsx"
    output_path = os.path.join(_user_storage_dir(user_id), output_filename)
    combined_df.to_excel(output_path, index=False)

    store_prepare_tp_final_file(user_id, output_path)

    return send_file(
        output_path,
        as_attachment=True,
        download_name=output_filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@prepare_tp_bp.route('/reset', methods=['POST'])
@login_required
def reset_prepare_tp():
    user_id = current_user.id
    records = _get_file_records(user_id)
    for record in records:
        for path_key in ['path', 'committed_path']:
            _safe_remove(record.get(path_key))

    _store_file_records(user_id, [])
    clear_prepare_tp_final_file(user_id)

    return jsonify({'success': True, 'message': 'Prepare TP session reset.'})


