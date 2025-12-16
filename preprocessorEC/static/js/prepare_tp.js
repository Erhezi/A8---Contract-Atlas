(() => {
	const configElement = document.getElementById('prepare-tp-config-data');
	if (!configElement) {
		console.warn('Prepare TP configuration not found.');
		return;
	}

	let config;
	try {
		config = JSON.parse(configElement.textContent);
	} catch (error) {
		console.error('Invalid Prepare TP configuration payload.', error);
		return;
	}

	const endpoints = config.endpoints || {};
	const state = {
		files: [],
	};
	const contractSuggestionTimers = new Map();
	const contractSuggestionRequests = new Map();

	const tableBody = document.querySelector('#prepare-tp-table tbody');
	const uploadInput = document.getElementById('prepare-tp-upload-input');
	const uploadButton = document.getElementById('prepare-tp-upload-btn');
	const browseButton = document.getElementById('prepare-tp-browse-btn');
	const uploadLabel = document.querySelector('label[for="prepare-tp-upload-input"]');
	const prepareButton = document.getElementById('prepare-tp-prepare-btn');
	const resetButton = document.getElementById('prepare-tp-reset-btn');

	const notify = (type, message) => {
		if (typeof showGlobalAlert === 'function') {
			showGlobalAlert(type, message);
		} else {
			// Fallback for environments without global alert helper
			alert(`${type.toUpperCase()}: ${message}`); // eslint-disable-line no-alert
		}
	};

	const endpoint = (name, fileId) => {
		const template = endpoints[name];
		if (!template) {
			throw new Error(`Endpoint not configured: ${name}`);
		}
		if (fileId) {
			return template.replace('__ID__', encodeURIComponent(fileId));
		}
		return template;
	};

	const getTomorrowDateString = () => {
		const tomorrow = new Date();
		tomorrow.setDate(tomorrow.getDate() + 1);
		return tomorrow.toISOString().split('T')[0];
	};

	const getGlobalSuggestionsContainer = () => {
		let container = document.querySelector('body > .prepare-tp-contract-suggestions');
		if (!container) {
			container = document.createElement('div');
			container.className = 'prepare-tp-contract-suggestions';
			document.body.appendChild(container);
		}
		return container;
	};

	const getContractSuggestionContainer = (input) => {
		return getGlobalSuggestionsContainer();
	};

	const escapeHtml = (value) => (
		(value || '').toString().replace(/[&<>"']/g, (char) => ({
			'&': '&amp;',
			'<': '&lt;',
			'>': '&gt;',
			'"': '&quot;',
			"'": '&#39;',
		}[char] || char))
	);

	const truncateText = (text, maxLength = 50) => {
		if (!text) {
			return '';
		}
		const trimmed = text.trim();
		if (trimmed.length <= maxLength) {
			return trimmed;
		}
		return `${trimmed.slice(0, maxLength - 3).trimEnd()}...`;
	};

	const buildContractSuggestionLabel = (item) => {
		const contract = escapeHtml(item.contract_number || '');
		const manufacturer = escapeHtml(item.manufacturer_name || '');
		const description = escapeHtml(truncateText(item.contract_description, 50));
		const parts = [contract];
		if (manufacturer) {
			parts.push(manufacturer);
		}
		if (description) {
			parts.push(description);
		}
		return parts.filter(Boolean).join(' | ');
	};

	const closeContractSuggestionContainer = (container) => {
		if (!container) {
			return;
		}
		const { fileId } = container.dataset;
		if (fileId) {
			if (contractSuggestionTimers.has(fileId)) {
				clearTimeout(contractSuggestionTimers.get(fileId));
				contractSuggestionTimers.delete(fileId);
			}
			contractSuggestionRequests.delete(fileId);
		}
		container.innerHTML = '';
		container.classList.remove('open');
	};

	const closeAllContractSuggestionContainers = () => {
		document.querySelectorAll('.prepare-tp-contract-suggestions.open').forEach((container) => {
			closeContractSuggestionContainer(container);
		});
	};

	const setContractSuggestions = (container, suggestions, input) => {
		if (!container) {
			return;
		}
		container.innerHTML = '';
		suggestions.forEach((item) => {
			if (!item?.contract_number) {
				return;
			}
			const button = document.createElement('button');
			button.type = 'button';
			button.className = 'prepare-tp-contract-suggestion';
			button.dataset.value = item.contract_number;
			button.dataset.fileId = container.dataset.fileId || '';
			button.title = item.contract_description || '';
			const label = buildContractSuggestionLabel(item);
			const separatorIndex = label.indexOf('|');
			if (separatorIndex === -1) {
				button.innerHTML = `<strong>${label}</strong>`;
			} else {
				const first = label.slice(0, separatorIndex).trim();
				const rest = label.slice(separatorIndex + 1).trim();
				button.innerHTML = `<strong>${first}</strong> | ${rest}`;
			}
			container.appendChild(button);
		});
		if (container.childElementCount) {
			container.classList.add('open');
			container.scrollTop = 0;
			if (input) {
				const rect = input.getBoundingClientRect();
				const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
				const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
				container.style.top = `${rect.bottom + scrollTop}px`;
				container.style.left = `${rect.left + scrollLeft}px`;
				container.style.width = `${rect.width}px`;
			}
		} else {
			container.classList.remove('open');
		}
	};

	const fetchContractSuggestions = async (query) => {
		if (!query || query.length < 3) {
			return [];
		}
		try {
			const url = new URL(endpoint('activeContractsSearch'), window.location.origin);
			url.searchParams.set('q', query);
			const response = await fetch(url.toString(), { credentials: 'same-origin' });
			const data = await response.json().catch(() => null);
			if (!response.ok || !data || !data.success) {
				throw new Error(data?.message || 'Failed to load contract suggestions.');
			}
			return Array.isArray(data.results) ? data.results : [];
		} catch (error) {
			console.error(error);
			return [];
		}
	};

	const requestContractSuggestions = (fileId, query, container, input) => {
		if (!fileId || !container || !input) {
			return;
		}
		if (contractSuggestionTimers.has(fileId)) {
			clearTimeout(contractSuggestionTimers.get(fileId));
		}
		if (!query || query.length < 3) {
			closeContractSuggestionContainer(container);
			return;
		}

		container.dataset.fileId = fileId;

		const timer = setTimeout(async () => {
			contractSuggestionTimers.delete(fileId);
			contractSuggestionRequests.set(fileId, query);
			const suggestions = await fetchContractSuggestions(query);
			if (contractSuggestionRequests.get(fileId) !== query) {
				return;
			}
			if (container.dataset.fileId !== fileId) {
				return;
			}
			if (!document.body.contains(input)) {
				return;
			}
			setContractSuggestions(container, suggestions, input);
		}, 250);
		contractSuggestionTimers.set(fileId, timer);
	};

	const isCommitReady = (file) => {
		const metadata = file.metadata || {};
		return Boolean(
			metadata.intended_action &&
			metadata.contract_number &&
			metadata.vendor_erp_id &&
			metadata.source_contract_type
		);
	};

	const findFile = (fileId) => state.files.find((file) => file.id === fileId);

	const removeFile = (fileId) => {
		const index = state.files.findIndex((file) => file.id === fileId);
		if (index >= 0) {
			state.files.splice(index, 1);
		}
	};

	const upsertFile = (updatedFile) => {
		const index = state.files.findIndex((file) => file.id === updatedFile.id);
		if (index >= 0) {
			state.files[index] = updatedFile;
		} else {
			state.files.push(updatedFile);
		}
	};

	const clearTable = () => {
		while (tableBody.firstChild) {
			tableBody.removeChild(tableBody.firstChild);
		}
	};

	const closeAllOrgDropdowns = () => {
		document.querySelectorAll('.prepare-tp-org-dropdown.open').forEach((el) => {
			el.classList.remove('open');
			const menu = el.querySelector('.prepare-tp-org-menu');
			if (menu) {
				menu.style.position = '';
				menu.style.top = '';
				menu.style.left = '';
				menu.style.minWidth = '';
			}
		});
	};

	const updateFooterButtons = () => {
		const anyCommitted = state.files.some((file) => file.status === 'committed');
		prepareButton.disabled = !anyCommitted;
		resetButton.disabled = state.files.length === 0;
	};

	const updateUploadButtonState = () => {
		if (!uploadButton) {
			return;
		}
		const hasFile = Boolean(uploadInput?.files && uploadInput.files.length > 0);
		uploadButton.disabled = !hasFile;
	};

	const resetUploadControl = () => {
		if (uploadInput) {
			uploadInput.value = '';
		}
		if (uploadLabel) {
			uploadLabel.textContent = 'Choose file';
		}
		updateUploadButtonState();
	};

	const hasDuplicateFilename = (filename) => {
		const normalized = filename.trim().toLowerCase();
		return state.files.some((file) => (file.filename || '').trim().toLowerCase() === normalized);
	};

	const normalizeOrganizationValues = (value) => {
		if (Array.isArray(value)) {
			return value.filter((item) => typeof item === 'string' && item.trim()).map((item) => item.trim());
		}
		if (typeof value === 'string') {
			return value
				.split(',')
				.map((item) => item.trim())
				.filter(Boolean);
		}
		return [];
	};

	const getOrganizationOptions = () => Array.isArray(config.organizations) ? config.organizations : [];

	const sameOrgSelection = (a, b) => {
		const normA = normalizeOrganizationValues(a).slice().sort();
		const normB = normalizeOrganizationValues(b).slice().sort();
		if (normA.length !== normB.length) return false;
		return normA.every((val, idx) => val === normB[idx]);
	};

	const applyOrganizationSelectionRules = (container) => {
		const checkboxes = Array.from(container.querySelectorAll('.prepare-tp-org-checkbox'));
		const mhsBox = checkboxes.find((cb) => cb.value === 'MHS');
		const hasMhs = Boolean(mhsBox?.checked);

		checkboxes.forEach((cb) => {
			if (cb.value === 'MHS') {
				cb.disabled = false;
				return;
			}
			cb.disabled = hasMhs;
			if (hasMhs) {
				cb.checked = false;
			}
		});
	};

	const getOrganizationSelection = (container) => (
		Array.from(container.querySelectorAll('.prepare-tp-org-checkbox'))
			.filter((cb) => cb.checked)
			.map((cb) => cb.value)
	);

	const updateOrganizationDisplay = (container) => {
		const button = container.querySelector('.prepare-tp-org-toggle');
		if (!button) return;
		const selected = getOrganizationSelection(container);
		button.textContent = selected.length ? selected.join(', ') : 'Select...';
	};

	const buildStatusCell = (file) => {
		const cell = document.createElement('td');
		cell.className = 'filename-cell';
		cell.title = `${file.row_count || 0} rows`;

		const wrapper = document.createElement('div');
		wrapper.className = 'prepare-tp-status';

		const nameSpan = document.createElement('span');
		nameSpan.className = 'prepare-tp-filename';
		nameSpan.textContent = file.filename;
		nameSpan.title = file.filename;
		wrapper.appendChild(nameSpan);

		if (file.status === 'committed') {
			const badge = document.createElement('span');
			badge.classList.add('prepare-tp-badge', 'committed');
			badge.textContent = 'Committed';
			wrapper.appendChild(badge);
		}
		cell.appendChild(wrapper);
		return cell;
	};

	const buildSelectCell = (file, fieldName, options) => {
		const cell = document.createElement('td');
		const select = document.createElement('select');
		select.className = 'prepare-tp-inline-select';
		select.dataset.fileId = file.id;
		select.dataset.field = fieldName;
		const placeholder = document.createElement('option');
		placeholder.value = '';
		placeholder.textContent = 'Select...';
		select.appendChild(placeholder);
		options.forEach((optionValue) => {
			const option = document.createElement('option');
			option.value = optionValue;
			option.textContent = optionValue;
			select.appendChild(option);
		});
		const metadata = file.metadata || {};
		select.value = metadata[fieldName] || '';
		cell.appendChild(select);
		return cell;
	};

	const buildOrganizationCell = (file) => {
		const cell = document.createElement('td');
		const container = document.createElement('div');
		container.className = 'prepare-tp-org-dropdown';
		container.dataset.fileId = file.id;

		const toggle = document.createElement('button');
		toggle.type = 'button';
		toggle.className = 'prepare-tp-org-toggle';
		toggle.textContent = 'Select organizations';
		container.appendChild(toggle);

		const menu = document.createElement('div');
		menu.className = 'prepare-tp-org-menu';
		const options = getOrganizationOptions();
		const sortedOptions = [
			...options.filter((opt) => opt.value === 'MHS'),
			...options.filter((opt) => opt.value !== 'MHS'),
		];

		sortedOptions.forEach((opt) => {
			const optionRow = document.createElement('label');
			optionRow.className = 'prepare-tp-org-option';
			const checkbox = document.createElement('input');
			checkbox.type = 'checkbox';
			checkbox.value = opt.value;
			checkbox.dataset.field = 'organization';
			checkbox.dataset.fileId = file.id;
			checkbox.className = 'prepare-tp-org-checkbox';
			const text = document.createElement('span');
			text.textContent = opt.label || opt.value;
			optionRow.appendChild(checkbox);
			optionRow.appendChild(text);
			menu.appendChild(optionRow);
		});

		container.appendChild(menu);
		cell.appendChild(container);

		const selectedOrgs = normalizeOrganizationValues(file.metadata?.organization);
		const checkboxes = container.querySelectorAll('.prepare-tp-org-checkbox');
		checkboxes.forEach((cb) => {
			if (selectedOrgs.includes(cb.value)) {
				cb.checked = true;
			}
		});
		applyOrganizationSelectionRules(container);
		updateOrganizationDisplay(container);
		return cell;
	};

	const buildInputCell = (file, fieldName, type = 'text') => {
		const cell = document.createElement('td');
		const input = document.createElement('input');
		input.type = type;
		input.className = 'prepare-tp-inline-input';
		input.dataset.fileId = file.id;
		input.dataset.field = fieldName;
		if (type === 'date') {
			const value = file[fieldName] || (file.metadata?.[fieldName]) || '';
			input.value = (value || '').substring(0, 10);
		} else {
			input.value = (file.metadata?.[fieldName] || '').toString();
		}

		let parent = cell;
		if (fieldName === 'contract_number' && type !== 'date') {
			const wrapper = document.createElement('div');
			wrapper.className = 'prepare-tp-contract-input-wrapper';
			wrapper.dataset.fileId = file.id;
			wrapper.appendChild(input);
			// Global suggestions container used instead
			cell.appendChild(wrapper);
			parent = wrapper;
			input.autocomplete = 'off';
			input.classList.add('prepare-tp-contract-input');
		} else {
			cell.appendChild(input);
		}
		return cell;
	};

	const createFileRow = (file) => {
		const row = document.createElement('tr');
		row.dataset.fileId = file.id;
		const metadata = file.metadata || {};
		const isNewContract = metadata.intended_action === 'New Contract';

		row.appendChild(buildStatusCell(file));
		row.appendChild(buildOrganizationCell(file));
		row.appendChild(buildSelectCell(file, 'intended_action', config.intendedActions || []));
		row.appendChild(buildInputCell(file, 'contract_number'));
		row.appendChild(buildInputCell(file, 'vendor_erp_id'));
		row.appendChild(buildSelectCell(file, 'source_contract_type', config.sourceTypes || []));

		const endDateCell = document.createElement('td');
		const endDateInput = document.createElement('input');
		endDateInput.type = 'date';
		endDateInput.className = 'prepare-tp-inline-input';
		endDateInput.dataset.fileId = file.id;
		endDateInput.dataset.field = 'current_contract_end_date';
		endDateInput.value = (file.current_contract_end_date || '').substring(0, 10);
		if (isNewContract) {
			endDateInput.min = getTomorrowDateString();
		}
		endDateCell.appendChild(endDateInput);
		row.appendChild(endDateCell);

		const actionsCell = document.createElement('td');
		const actionsWrapper = document.createElement('div');
		actionsWrapper.className = 'prepare-tp-actions';

		const fetchButton = document.createElement('button');
		fetchButton.type = 'button';
		fetchButton.className = 'fetch';
		fetchButton.dataset.action = 'fetch';
		fetchButton.textContent = 'Fetch';
		fetchButton.disabled = isNewContract || !metadata.contract_number;

		const commitButton = document.createElement('button');
		commitButton.type = 'button';
		commitButton.className = 'commit';
		commitButton.dataset.action = 'commit';
		const commitIcon = document.createElement('i');
		commitIcon.className = 'fas fa-check';
		commitButton.appendChild(commitIcon);
		commitButton.title = 'Commit file';
		commitButton.setAttribute('aria-label', 'Commit file');
		// add single-letter label to save space
		const commitLabel = document.createElement('span');
		commitLabel.className = 'action-icon-label';
		commitLabel.textContent = '✓';
		commitButton.appendChild(commitLabel);
		commitButton.disabled = file.status === 'committed' || !isCommitReady(file);

		const removeButton = document.createElement('button');
		removeButton.type = 'button';
		removeButton.className = 'remove';
		removeButton.dataset.action = 'remove';
		const removeIcon = document.createElement('i');
		removeIcon.className = 'fas fa-times';
		removeButton.appendChild(removeIcon);
		removeButton.title = 'Remove file';
		removeButton.setAttribute('aria-label', 'Remove file');
		// add single-letter label to save space
		const removeLabel = document.createElement('span');
		removeLabel.className = 'action-icon-label';
		removeLabel.textContent = '✗';
		removeButton.appendChild(removeLabel);

		actionsWrapper.append(fetchButton, commitButton, removeButton);
		actionsCell.appendChild(actionsWrapper);
		row.appendChild(actionsCell);

		return row;
	};

	const refreshRow = (file) => {
		const row = tableBody.querySelector(`tr[data-file-id="${file.id}"]`);
		if (!row) {
			renderTable();
			return;
		}

		const metadata = file.metadata || {};
		const isNewContract = metadata.intended_action === 'New Contract';
		const statusCell = row.querySelector('.prepare-tp-status');
		if (statusCell) {
			statusCell.title = `${file.row_count || 0} rows`;
			const filenameSpan = statusCell.querySelector('.prepare-tp-filename');
			if (filenameSpan) {
				filenameSpan.textContent = file.filename;
				filenameSpan.title = file.filename;
			}
			let badge = statusCell.querySelector('.prepare-tp-badge');
			if (file.status === 'committed') {
				if (!badge) {
					badge = document.createElement('span');
					badge.classList.add('prepare-tp-badge', 'committed');
					statusCell.appendChild(badge);
				}
				badge.className = 'prepare-tp-badge committed';
				badge.textContent = 'Committed';
			} else if (badge) {
				badge.remove();
			}
		}

		const intendedSelect = row.querySelector('select[data-field="intended_action"]');
		if (intendedSelect) {
			intendedSelect.value = metadata.intended_action || '';
		}

		const orgContainer = row.querySelector('.prepare-tp-org-dropdown');
		if (orgContainer) {
			const selectedOrgs = normalizeOrganizationValues(metadata.organization);
			Array.from(orgContainer.querySelectorAll('.prepare-tp-org-checkbox')).forEach((cb) => {
				cb.checked = selectedOrgs.includes(cb.value);
			});
			applyOrganizationSelectionRules(orgContainer);
			updateOrganizationDisplay(orgContainer);
		}

		const contractInput = row.querySelector('input[data-field="contract_number"]');
		if (contractInput) {
			contractInput.value = metadata.contract_number || '';
		}

		const vendorInput = row.querySelector('input[data-field="vendor_erp_id"]');
		if (vendorInput) {
			vendorInput.value = metadata.vendor_erp_id || '';
		}

		const sourceSelect = row.querySelector('select[data-field="source_contract_type"]');
		if (sourceSelect) {
			sourceSelect.value = metadata.source_contract_type || '';
		}

		const endDateInput = row.querySelector('input[data-field="current_contract_end_date"]');
		if (endDateInput) {
			endDateInput.value = (file.current_contract_end_date || '').substring(0, 10);
			if (isNewContract) {
				endDateInput.min = getTomorrowDateString();
			} else {
				endDateInput.removeAttribute('min');
			}
		}

		const fetchButton = row.querySelector('button[data-action="fetch"]');
		if (fetchButton) {
			fetchButton.disabled = isNewContract || !metadata.contract_number;
		}

		const commitButton = row.querySelector('button[data-action="commit"]');
		if (commitButton) {
			commitButton.disabled = file.status === 'committed' || !isCommitReady(file);
		}
	};

	const renderTable = () => {
		clearTable();

		if (state.files.length) {
			state.files.forEach((file) => {
				tableBody.appendChild(createFileRow(file));
			});
		} else {
			const emptyRow = document.createElement('tr');
			emptyRow.className = 'prepare-tp-empty';
			const cell = document.createElement('td');
			cell.colSpan = 8;
			cell.textContent = 'No files uploaded yet. Add your first contract file to begin.';
			emptyRow.appendChild(cell);
			tableBody.appendChild(emptyRow);
		}

		updateFooterButtons();
	};

	const loadState = async () => {
		try {
			const response = await fetch(endpoint('state'), { credentials: 'same-origin' });
			if (!response.ok) {
				throw new Error('Failed to load state');
			}
			const data = await response.json();
			if (!data.success) {
				throw new Error(data.message || 'Unable to load files');
			}
			if (Array.isArray(data.organizations)) {
				config.organizations = data.organizations;
			}
			state.files = data.files || [];
			renderTable();
		} catch (error) {
			console.error(error);
			notify('danger', 'Unable to load Prepare TP state.');
		}
	};

	const sendMetadata = async (fileId, payload) => {
		const response = await fetch(endpoint('metadata', fileId), {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			credentials: 'same-origin',
			body: JSON.stringify(payload),
		});
		const data = await response.json().catch(() => null);
		if (!response.ok || !data || !data.success) {
			throw new Error(data?.message || 'Failed to update metadata.');
		}
		upsertFile(data.file);
		refreshRow(data.file);
		updateFooterButtons();
		return data;
	};

	const handleUpload = async () => {
		if (!uploadInput || !uploadButton) {
			return;
		}

		const file = uploadInput.files?.[0];
		if (!file) {
			notify('warning', 'Please choose a file to upload.');
			updateUploadButtonState();
			return;
		}

		const originalName = (file.name || '').trim();
		if (!originalName) {
			notify('warning', 'Selected file name is invalid. Please choose a different file.');
			resetUploadControl();
			return;
		}

		if (!originalName.toLowerCase().endsWith('.xlsx')) {
			notify('warning', 'Only .xlsx files are supported.');
			resetUploadControl();
			return;
		}

		if (hasDuplicateFilename(originalName)) {
			notify('warning', 'That file has already been uploaded.');
			return;
		}

		const formData = new FormData();
		formData.append('file', file);
		uploadButton.disabled = true;

		try {
			const response = await fetch(endpoint('upload'), {
				method: 'POST',
				body: formData,
				credentials: 'same-origin',
			});
			const data = await response.json().catch(() => null);
			if (!response.ok || !data || !data.success) {
				throw new Error(data?.message || 'File upload failed.');
			}
			upsertFile(data.file);
			renderTable();
			notify('success', data.message || 'File uploaded successfully.');
			resetUploadControl();
		} catch (error) {
			console.error(error);
			notify('danger', error.message || 'Unable to upload file.');
		} finally {
			updateUploadButtonState();
		}
	};

	const handleFetch = async (fileId, button) => {
		const file = findFile(fileId);
		if (!file) {
			notify('danger', 'File not found.');
			return;
		}

		const contractNumber = file.metadata?.contract_number?.trim();
		if (!contractNumber) {
			notify('warning', 'Add a contract number before fetching details.');
			const input = tableBody.querySelector(`input.prepare-tp-inline-input[data-file-id="${fileId}"][data-field="contract_number"]`);
			input?.focus();
			return;
		}

		button.disabled = true;
		try {
			const url = new URL(endpoint('fetchContract'), window.location.origin);
			url.searchParams.set('contract_number', contractNumber);
			const response = await fetch(url.toString(), { credentials: 'same-origin' });
			const data = await response.json().catch(() => null);
			if (!response.ok || !data || !data.success) {
				throw new Error(data?.message || 'Unable to fetch contract details.');
			}

			const fetched = data.data || {};
			const payload = {
				intended_action: file.metadata?.intended_action || null,
				contract_number: fetched.contract_number || contractNumber,
				vendor_erp_id: fetched.vendor_erp_id || file.metadata?.vendor_erp_id || '',
				source_contract_type: fetched.source_contract_type || file.metadata?.source_contract_type || '',
				current_contract_end_date: fetched.current_contract_end_date || null,
			};

			await sendMetadata(fileId, payload);
			notify('success', data.message || 'Contract details fetched.');
		} catch (error) {
			console.error(error);
			notify('danger', error.message || 'Failed to fetch contract details.');
		} finally {
			button.disabled = false;
		}
	};

	const handleCommit = async (fileId, button) => {
		button.disabled = true;
		try {
			const response = await fetch(endpoint('commit', fileId), {
				method: 'POST',
				credentials: 'same-origin',
			});
			const data = await response.json().catch(() => null);
			if (!response.ok || !data || !data.success) {
				throw new Error(data?.message || 'Failed to commit file.');
			}
			upsertFile(data.file);
			refreshRow(data.file);
			updateFooterButtons();
			notify('success', data.message || 'File committed successfully.');
		} catch (error) {
			console.error(error);
			notify('danger', error.message || 'Failed to commit file.');
			button.disabled = false;
		}
	};

	const handleRemove = async (fileId, button) => {
		if (!window.confirm('Remove this file from Prepare TP? This will delete the uploaded data.')) { // eslint-disable-line no-alert
			return;
		}

		button.disabled = true;
		try {
			const response = await fetch(endpoint('remove', fileId), {
				method: 'DELETE',
				credentials: 'same-origin',
			});
			const data = await response.json().catch(() => null);
			if (!response.ok || !data || !data.success) {
				throw new Error(data?.message || 'Failed to remove file.');
			}

			if (Array.isArray(data.files)) {
				state.files = data.files;
			} else {
				removeFile(fileId);
			}

			renderTable();
			notify('info', data.message || 'File removed.');
		} catch (error) {
			console.error(error);
			notify('danger', error.message || 'Failed to remove file.');
			if (document.body.contains(button)) {
				button.disabled = false;
			}
		}
	};

	const handlePrepare = async () => {
		prepareButton.disabled = true;
		try {
			const response = await fetch(endpoint('prepareOutput'), {
				method: 'POST',
				credentials: 'same-origin',
			});

			const contentType = response.headers.get('Content-Type') || '';
			if (!response.ok || contentType.includes('application/json')) {
				const data = await response.json().catch(() => null);
				throw new Error(data?.message || 'Unable to prepare TP file.');
			}

			const blob = await response.blob();
			const disposition = response.headers.get('Content-Disposition') || '';
			const filenameMatch = /filename="?([^";]+)"?/i.exec(disposition);
			const downloadName = filenameMatch ? filenameMatch[1] : 'prepared_tp.xlsx';

			const blobUrl = URL.createObjectURL(blob);
			const link = document.createElement('a');
			link.href = blobUrl;
			link.download = downloadName;
			document.body.appendChild(link);
			link.click();
			link.remove();
			URL.revokeObjectURL(blobUrl);

			notify('success', 'Prepared TP file downloaded.');
		} catch (error) {
			console.error(error);
			notify('danger', error.message || 'Unable to prepare TP file.');
		} finally {
			renderTable();
		}
	};

	const handleReset = async () => {
		if (!window.confirm('Reset the Prepare TP session? All uploaded data will be cleared.')) { // eslint-disable-line no-alert
			return;
		}
		resetButton.disabled = true;
		try {
			const response = await fetch(endpoint('reset'), {
				method: 'POST',
				credentials: 'same-origin',
			});
			const data = await response.json().catch(() => null);
			if (!response.ok || !data || !data.success) {
				throw new Error(data?.message || 'Unable to reset session.');
			}
			state.files = [];
			renderTable();
			resetUploadControl();
			notify('info', data.message || 'Session reset.');
		} catch (error) {
			console.error(error);
			notify('danger', error.message || 'Failed to reset session.');
		}
	};

	const handleContractInput = (event) => {
		const input = event.target;
		if (!input.classList.contains('prepare-tp-contract-input')) {
			return;
		}
		const fileId = input.dataset.fileId;
		if (!fileId) {
			return;
		}
		const container = getContractSuggestionContainer(input);
		if (!container) {
			return;
		}
		const query = input.value.trim();
		requestContractSuggestions(fileId, query, container, input);
	};

	const handleContractSuggestionClick = (event) => {
		const button = event.target.closest('.prepare-tp-contract-suggestion');
		if (!button) {
			return;
		}
		const { value, fileId } = button.dataset;
		if (!fileId) {
			return;
		}
		const input = tableBody?.querySelector(`.prepare-tp-contract-input[data-file-id="${fileId}"]`);
		const container = button.parentElement;
		closeContractSuggestionContainer(container);
		if (!input) {
			return;
		}
		input.value = value || '';
		input.dispatchEvent(new Event('change', { bubbles: true }));
		input.focus();
	};

	const handleInlineChange = async (event) => {
		const target = event.target;
		const isOrgCheckbox = target.classList.contains('prepare-tp-org-checkbox');
		if (!target.classList.contains('prepare-tp-inline-input') && !target.classList.contains('prepare-tp-inline-select') && !isOrgCheckbox) {
			return;
		}

		const fileId = target.dataset.fileId;
		const field = target.dataset.field;
		if (!fileId || !field) {
			return;
		}

		const file = findFile(fileId);
		if (!file) {
			notify('danger', 'File not found.');
			return;
		}

		let value = target.value;
		const payload = {};

		if (field === 'organization') {
			const container = target.closest('.prepare-tp-org-dropdown');
			if (!container) {
				return;
			}
			applyOrganizationSelectionRules(container);
			const normalized = normalizeOrganizationValues(getOrganizationSelection(container));
			const currentValue = file.metadata?.organization || [];
			if (sameOrgSelection(normalized, currentValue)) {
				return;
			}
			payload[field] = normalized.length ? normalized : null;
			updateOrganizationDisplay(container);
		} else {
			if (target.type !== 'date' && typeof value === 'string') {
				value = value.trim();
			}
			payload[field] = value === '' ? null : value;

			const currentValue = field === 'current_contract_end_date'
				? (file.current_contract_end_date || null)
				: (file.metadata?.[field] ?? null);
			if ((payload[field] || null) === (currentValue || null)) {
				target.value = payload[field] ?? '';
				return;
			}
		}

		target.disabled = true;
		try {
			await sendMetadata(fileId, payload);
			target.disabled = false;
		} catch (error) {
			console.error(error);
			notify('danger', error.message || 'Failed to update metadata.');
			target.disabled = false;
		}
	};

	const handleTableClick = (event) => {
		if (event.target.closest('.prepare-tp-contract-suggestion')) {
			return;
		}
		const button = event.target.closest('button[data-action]');
		const orgToggle = event.target.closest('.prepare-tp-org-toggle');
		if (orgToggle) {
			const container = orgToggle.closest('.prepare-tp-org-dropdown');
			if (container) {
				const alreadyOpen = container.classList.contains('open');
				closeAllOrgDropdowns();
				if (!alreadyOpen) {
					container.classList.add('open');
					const menu = container.querySelector('.prepare-tp-org-menu');
					if (menu) {
						const rect = orgToggle.getBoundingClientRect();
						menu.style.position = 'fixed';
						menu.style.top = `${rect.bottom + 4}px`;
						menu.style.left = `${rect.left}px`;
						menu.style.minWidth = `${rect.width}px`;
					}
				}
			}
			return;
		}

		if (!button) {
			return;
		}

		const row = button.closest('tr');
		const fileId = row?.dataset.fileId;
		if (!fileId) {
			notify('danger', 'Unable to determine which file to update.');
			return;
		}

		switch (button.dataset.action) {
		case 'fetch':
			handleFetch(fileId, button);
			break;
		case 'commit':
			handleCommit(fileId, button);
			break;
		case 'remove':
			handleRemove(fileId, button);
			break;
		default:
			console.warn('Unknown action:', button.dataset.action);
		}
	};

	const bindEvents = () => {
		uploadInput?.addEventListener('change', () => {
			updateUploadButtonState();
			if (uploadLabel && uploadInput.files && uploadInput.files.length > 0) {
				uploadLabel.textContent = uploadInput.files[0].name;
			} else if (uploadLabel) {
				uploadLabel.textContent = 'Choose file';
			}
		});
		browseButton?.addEventListener('click', () => {
			uploadInput?.click();
		});
		uploadButton?.addEventListener('click', handleUpload);
		prepareButton?.addEventListener('click', handlePrepare);
		resetButton?.addEventListener('click', handleReset);
		tableBody?.addEventListener('click', handleTableClick);
		// Moved to document level for global container
		// tableBody?.addEventListener('click', handleContractSuggestionClick);
		tableBody?.addEventListener('input', handleContractInput);
		tableBody?.addEventListener('change', handleInlineChange);
		document.addEventListener('click', handleContractSuggestionClick);
		document.addEventListener('click', (event) => {
			if (!event.target.closest('.prepare-tp-org-dropdown')) {
				closeAllOrgDropdowns();
			}
			if (!event.target.closest('.prepare-tp-contract-input-wrapper') && !event.target.closest('.prepare-tp-contract-suggestions')) {
				closeAllContractSuggestionContainers();
			}
		});
		document.addEventListener('scroll', (event) => {
			if (event.target && event.target.classList && event.target.classList.contains('prepare-tp-contract-suggestions')) {
				return;
			}
			closeAllOrgDropdowns();
			closeAllContractSuggestionContainers();
		}, { capture: true, passive: true });
		window.addEventListener('resize', () => {
			closeAllContractSuggestionContainers();
		}, { passive: true });
	};

	bindEvents();
	updateUploadButtonState();
	renderTable();
	loadState();
})();
