document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const goToExportBtn = document.getElementById('go-to-export-btn');
    const goHomeBtn = document.getElementById('go-home-btn');
    const deleteTaskBtns = document.querySelectorAll('.delete-task-btn');
    const loadingSpinner = document.getElementById('loading-spinner');

    const historySearchInput = document.getElementById('history-search');
    const historySearchType = document.getElementById('history-search-type');
    const clearHistorySearchBtn = document.getElementById('clear-history-search-btn');

    initContractLinking();
    initHoldTaskButtons();
    makeHistoryTableSortable();

    // Navigate to Export Function (Step 6)
    if (goToExportBtn) {
        goToExportBtn.addEventListener('click', function() {
            if (loadingSpinner) {
                loadingSpinner.style.display = 'flex';
            }
            window.location.href = getApiUrl('/goto-step/6');
        });
    }

    // Navigate to Home
    if (goHomeBtn) {
        goHomeBtn.addEventListener('click', function() {
            if (loadingSpinner) {
                loadingSpinner.style.display = 'flex';
            }
            window.location.href = getApiUrl('/');
        });
    }

    // Confirm delete task
    if (deleteTaskBtns) {
        deleteTaskBtns.forEach(function(btn) {
            btn.addEventListener('click', function(event) {
                if (!confirm('Are you sure you want to delete this task? this action can NOT be undone.')) {
                    event.preventDefault();
                    return false;
                }
                if (loadingSpinner) {
                    loadingSpinner.style.display = 'flex';
                }
                return true;
            });
        });
    }

    // Add event listeners
    if (historySearchInput) {
        historySearchInput.addEventListener('input', filterHistoryTable);
    }
    
    if (historySearchType) {
        historySearchType.addEventListener('change', filterHistoryTable);
    }
    
    if (clearHistorySearchBtn) {
        clearHistorySearchBtn.addEventListener('click', function() {
            historySearchInput.value = '';
            historySearchType.value = 'all';
            filterHistoryTable();
        });
    }

    function filterHistoryTable() {
        const searchTerm = historySearchInput.value.toLowerCase();
        const searchType = historySearchType.value;
        const tableRows = document.querySelectorAll('#task-history-table tbody tr');
        
        tableRows.forEach(row => {
            if (!searchTerm) {
                row.style.display = '';
                return;
            }
            
            let match = false;
            if (searchType === 'all') {
                // Search in all cells
                const rowText = row.textContent.toLowerCase();
                match = rowText.includes(searchTerm);
            } else {
                // Get column index based on search type
                let columnIndex = 0;
                switch (searchType) {
                    case 'task-id': columnIndex = 0; break;
                    case 'user': columnIndex = 1; break;
                    case 'wrike-id': columnIndex = 2; break;
                    case 'filename': columnIndex = 3; break;
                    case 'status': columnIndex = 5; break; // Using Status (CCX) column
                    case 'status2': columnIndex = 6; break; // Using Status (Infor) column
                }
                
                const cell = row.cells[columnIndex];
                if (cell) {
                    const cellText = cell.textContent.toLowerCase();
                    match = cellText.includes(searchTerm);
                }
            }
            
            row.style.display = match ? '' : 'none';
        });
    }

    function makeHistoryTableSortable() {
        const table = document.getElementById('task-history-table');
        if (!table) return;

        const thead = table.querySelector('thead');
        const tbody = table.querySelector('tbody');
        const headers = Array.from(thead.querySelectorAll('th'));

        headers.forEach((th, index) => {
            th.classList.add('sortable');
            const indicator = document.createElement('span');
            indicator.className = 'sort-indicator';
            indicator.textContent = '';
            th.appendChild(indicator);

            th.addEventListener('click', function () {
                const currentDir = th.dataset.sortDir === 'asc' ? 'desc' : 'asc';
                //clear sort direction for all headers
                headers.forEach(h => { delete h.dataset.sortDir; h.querySelector('.sort-indicator').textContent = ''; });
                th.dataset.sortDir = currentDir;
                th.querySelector('.sort-indicator').textContent = currentDir === 'asc' ? '▲' : '▼';

                // perform sorting
                const rows = Array.from(tbody.querySelectorAll('tr'));
                rows.sort((a, b) => compareTableRows(a, b, index, currentDir === 'asc'));
                // reappend sorted rows
                rows.forEach(row => tbody.appendChild(row));
            });
        });
    }

    function compareTableRows(rowA, rowB, colIndex,asc = true) {
        const cellA = (rowA.cells[colIndex] ? rowA.cells[colIndex].textContent.trim() : '').toLowerCase();
        const cellB = (rowB.cells[colIndex] ? rowB.cells[colIndex].textContent.trim() : '').toLowerCase();

        // datetime
        const dateA = parseDateString(cellA);
        const dateB = parseDateString(cellB);
        if (dateA && dateB) {
            return asc ? dateA - dateB : dateB - dateA;
        }
        
        // string (alsways fall back to string comparison)
        return asc ? cellA.localeCompare(cellB, undefined, { numeric: true, sensitivity: 'base' }) :
                     cellB.localeCompare(cellA, undefined, { numeric: true, sensitivity: 'base' });
    }

    function parseDateString(s) {
        // accept format like '2023-10-05 14:30:00'
        const m = s.match(/^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})/);
        if (m) {
            return new Date(m[1], m[2] - 1, m[3], m[4], m[5], m[6]).getTime();
        }
        return null;
    }

    function initContractLinking() {
        const commitButtons = document.querySelectorAll('.commit-link-btn');
        const resetButtons = document.querySelectorAll('.reset-link-btn');
        const erpVendorInputs = document.querySelectorAll('.erp-vendor-id-ccx');
        const contractNumberInputs = document.querySelectorAll('.contract-number');

        if (!commitButtons.length) return;

        // Automatically format Contract Number (CCX Sync) inputs
        contractNumberInputs.forEach(input => {
            input.addEventListener('input', function () {
                // Convert to uppercase and trim leading/trailing spaces
                this.value = this.value.toUpperCase().trim();
            });
        });

        // Check for already linked contracts and disable inputs for 'Batch Upload'
        document.querySelectorAll('#contract-linking-table tbody tr').forEach(row => {
            const exportGroup = row.querySelector('.export-group').textContent.trim();
            const contractInput = row.querySelector('.contract-number');
            const erpVendorInput = row.querySelector('.erp-vendor-id-ccx');
            const commitBtn = row.querySelector('.commit-link-btn');

            // Disable Contract Number (CCX Sync) input if Export Group is 'Batch Upload'
            if (exportGroup === 'Batch Upload') {
                contractInput.disabled = true;
                contractInput.style.backgroundColor = '#e9ecef'; // Add a disabled styling
                contractInput.setAttribute('data-original-value', contractInput.value); // Store the original value
            }

            // If both inputs have values, mark as linked
            if (contractInput && erpVendorInput && 
                contractInput.value.trim() !== '' && 
                erpVendorInput.value.trim() !== '') {
                
                // Apply linked styling
                contractInput.disabled = true;
                erpVendorInput.disabled = true;
                contractInput.style.backgroundColor = '#e9ecef';
                erpVendorInput.style.backgroundColor = '#e9ecef';
                commitBtn.disabled = true;
                commitBtn.textContent = 'Linked';
                row.classList.add('table-success');
            }
        });

        // Add ERP Vendor ID validation
        erpVendorInputs.forEach(input => {
            input.addEventListener('input', validateErpVendorId);
            // Initial validation
            validateErpVendorId.call(input);
        });

        // Reset button functionality
        resetButtons.forEach(button => {
            button.addEventListener('click', function () {
                const row = this.closest('tr');
                const exportGroup = row.querySelector('.export-group').textContent.trim();
                const contractInput = row.querySelector('.contract-number');
                const erpVendorInput = row.querySelector('.erp-vendor-id-ccx');
                const commitBtn = row.querySelector('.commit-link-btn');

                // Reset values
                if (exportGroup === 'Batch Upload') {
                    // Restore the original value for Batch Upload rows
                    const originalValue = contractInput.getAttribute('data-original-value') || '';
                    contractInput.value = originalValue;
                } else {
                    // Clear the value for other rows
                    contractInput.value = '';
                    contractInput.disabled = false;
                    contractInput.style.backgroundColor = '';
                }

                erpVendorInput.value = '';
                erpVendorInput.disabled = false;
                erpVendorInput.style.backgroundColor = '';

                // Reset commit button
                commitBtn.disabled = false;
                commitBtn.textContent = 'Commit';

                // Remove success styling
                row.classList.remove('table-success');

                // Reset validation
                erpVendorInput.style.borderColor = '';
                const feedback = erpVendorInput.nextElementSibling;
                if (feedback && feedback.classList.contains('invalid-feedback')) {
                    feedback.style.display = 'none';
                }
            });
        });

        commitButtons.forEach(button => {
            button.addEventListener('click', function () {
                const row = this.closest('tr');

                // Get data from the row
                const taskId = row.querySelector('.task-id').textContent.trim();
                const exportGroup = row.querySelector('.export-group').textContent.trim();
                const contractNumberPrp = row.querySelector('.contract-number-prp').textContent.trim();
                const erpVendorIdPrp = row.querySelector('.erp-vendor-id-prp').textContent.trim();
                const contractNumber = row.querySelector('.contract-number').value.trim();
                const erpVendorIdCcx = row.querySelector('.erp-vendor-id-ccx').value.trim();

                // Validate inputs
                if (!contractNumber && exportGroup !== 'Batch Upload') {
                    alert('Contract Number (CCX Sync) is required.');
                    return;
                }
                if (!erpVendorIdCcx) {
                    alert('ERP Vendor ID (CCX Sync) is required.');
                    return;
                }

                // Validate ERP Vendor ID format
                if (!isValidErpVendorId(erpVendorIdCcx)) {
                    alert('ERP Vendor ID (CCX Sync) must be 7 digits or 7 digits-B000.');
                    return;
                }

                // Show loading spinner
                if (loadingSpinner) {
                    loadingSpinner.style.display = 'flex';
                }

                // Send API request
                fetch(getApiUrl('/data-export/commit-contract-link'), {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        task_id: taskId,
                        export_group: exportGroup,
                        contract_number_prp: contractNumberPrp,
                        erp_vendor_id_prp: erpVendorIdPrp,
                        contract_number: contractNumber,
                        erp_vendor_id_ccx: erpVendorIdCcx
                    })
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading spinner
                    if (loadingSpinner) {
                        loadingSpinner.style.display = 'none';
                    }

                    if (data.success) {
                        // Show success message and disable inputs/button
                        alert(data.message);
                        const contractInput = row.querySelector('.contract-number');
                        const erpVendorInput = row.querySelector('.erp-vendor-id-ccx');

                        // Disable inputs
                        contractInput.disabled = true;
                        erpVendorInput.disabled = true;

                        // Apply committed styling
                        contractInput.style.backgroundColor = '#e9ecef';
                        erpVendorInput.style.backgroundColor = '#e9ecef';

                        // Disable commit button and update text
                        button.disabled = true;
                        button.textContent = 'Linked';

                        // Add success styling to row
                        row.classList.add('table-success');
                    } else {
                        alert(`Error: ${data.message}`);
                    }
                })
                .catch(error => {
                    // Hide loading spinner
                    if (loadingSpinner) {
                        loadingSpinner.style.display = 'none';
                    }
                    alert(`Error: ${error.message}`);
                });
            });
        });

        // Helper functions for ERP Vendor ID validation
        function isValidErpVendorId(value) {
            return /^[0-9]{7}(-B[0-9]{3})?$/.test(value);
        }

        function validateErpVendorId() {
            const value = this.value.trim();
            const isValid = isValidErpVendorId(value);
            const feedback = this.nextElementSibling;

            if (value === '') {
                this.style.borderColor = '';
                if (feedback && feedback.classList.contains('invalid-feedback')) {
                    feedback.style.display = 'none';
                }
                return;
            }

            if (isValid) {
                this.style.borderColor = '#28a745';
                if (feedback && feedback.classList.contains('invalid-feedback')) {
                    feedback.style.display = 'none';
                }
            } else {
                this.style.borderColor = '#dc3545';
                if (feedback && feedback.classList.contains('invalid-feedback')) {
                    feedback.style.display = 'block';
                }
            }

            // Control the commit button
            const row = this.closest('tr');
            const commitBtn = row.querySelector('.commit-link-btn');
            if (commitBtn) {
                commitBtn.disabled = !isValid && value !== '';
            }
        }
    }

    
    function initHoldTaskButtons() {
        const holdTaskBtns = document.querySelectorAll('.hold-task-btn');
        
        if (!holdTaskBtns.length) return;
        
        holdTaskBtns.forEach(btn => {
            btn.addEventListener('click', function() {
                const taskId = this.getAttribute('data-task-id');
                const currentStatus = this.getAttribute('data-status');
                const newStatus = currentStatus === 'Hold' ? 'Pending' : 'Hold';
                
                // Show loading spinner
                if (loadingSpinner) {
                    loadingSpinner.style.display = 'flex';
                }
                
                // Send API request
                fetch(getApiUrl('/data-export/toggle-task-hold'), {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        task_id: taskId,
                        new_status: newStatus
                    })
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading spinner
                    if (loadingSpinner) {
                        loadingSpinner.style.display = 'none';
                    }
                    
                    if (data.success) {
                        // Update button text and data attribute
                        this.textContent = newStatus === 'Hold' ? 'UH' : 'H';
                        this.setAttribute('data-status', newStatus);
                        
                        // Update status badge in the table
                        const statusCell = this.closest('tr').querySelector('td:nth-child(6)');
                        const statusBadge = statusCell.querySelector('.badge');
                        
                        if (statusBadge) {
                            statusBadge.textContent = newStatus;
                            
                            // Update badge class
                            if (newStatus === 'Hold') {
                                statusBadge.className = 'badge bg-progress-light';
                            } else {
                                statusBadge.className = 'badge bg-progress';
                            }
                        }
                        
                        // Toggle delete button state based on hold status
                        const row = this.closest('tr');
                        const deleteBtn = row.querySelector('.delete-task-btn');
                        if (deleteBtn) {
                            if (newStatus === 'Hold') {
                                deleteBtn.disabled = true;  // Disable delete when on hold
                            } else {
                                deleteBtn.disabled = false; // Enable delete when pending
                            }
                        }
                    } else {
                        alert('Failed to update task status: ' + data.message);
                    }
                })
                .catch(error => {
                    // Hide loading spinner
                    if (loadingSpinner) {
                        loadingSpinner.style.display = 'none';
                    }
                    alert('Error: ' + error.message);
                });
            });
        });
    }

});