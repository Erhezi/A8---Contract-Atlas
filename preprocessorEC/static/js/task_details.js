document.addEventListener('DOMContentLoaded', function() {

    const revertAllBtn = document.getElementById('revert-all-btn');

    // if revert all btton clicked, run revertAllEdits function
    if (revertAllBtn) {
        revertAllBtn.addEventListener('click', function() {
            revertAllEdits();
        });
    }

    // Check if taskId exists and load sync status data
    if (typeof taskId !== 'undefined') {
        loadSyncStatus();
    }

    // Check if taskId exists and if there are errors to load
    if (typeof taskId !== 'undefined' && document.getElementById('error-rows')) {
        loadTaskErrors();
    }

    // Function to load task errors
    function loadTaskErrors() {
        fetch(getApiUrl(`/data-export/task/${taskId}/errors`))
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    displayErrorRows(data.errors);
                } else {
                    showErrorMessage(data.message);
                }
            })
            .catch(error => {
                showErrorMessage(`Failed to load error data: ${error.message}`);
            });
    }

    // Function to display error rows in the table
    function displayErrorRows(errors) {
        const errorRows = document.getElementById('error-rows');
        
        // Clear loading message
        errorRows.innerHTML = '';
        
        if (errors.length === 0) {
            errorRows.innerHTML = '<tr><td colspan="16" class="text-center">No error records found.</td></tr>';
            return;
        }

        // check if action columns exist in the table (due to permission)
        const hasActionCols = document.querySelector('#task-errors-table th.action-col') !== null;

        errors.forEach(row => {
            const tr = document.createElement('tr');
            tr.setAttribute('data-pkid', row.PKID);
            tr.setAttribute('data-dataset', row.DataSet);
            tr.setAttribute('data-filerow', row.FileRow);
            tr.setAttribute('data-contract', row['Contract Number']);
            tr.setAttribute('data-mfg-part-num-bsl', row['Mfg Part Num']);
            tr.setAttribute('data-vendor-part-num-bsl', row['Vendor Part Num']);
            tr.setAttribute('data-uom-bsl', row.UOM);
            tr.setAttribute('data-qoe-bsl', row.QOE);


            // Apply row shading based on Group and Intended Action
            if (row.Group === 'Keep') {
                if (row['Intended Action'] === 'Expire') {
                    tr.classList.add('row-expire'); // Add a class for red shading
                } else if (row['Intended Action'] === 'Upsert') {
                    tr.classList.add('row-upsert'); // Add a class for green shading
                }
            }

            
            // Format dates for display
            const effectiveDate = row['Effective Date'] ? new Date(row['Effective Date']).toLocaleDateString() : '';
            const expirationDate = row['Expiration Date'] ? new Date(row['Expiration Date']).toLocaleDateString() : '';

            // Format the price for display
            const price = row['Contract Price'] ? `$${parseFloat(row['Contract Price']).toFixed(2)}` : '';
            
            // Create description with title for tooltip
            const description = row.Description || '';

            const validationFlag = row['File Row Validation Flag'] || '';
            const intendedAction = row['Intended Action'] || '';
            let formattedValidationFlag = validationFlag.split(' - ')[0]; // Get the part before the dash
            const afterDash = validationFlag.split(' - ')[1]; // Get the part after the dash, if it exists

            if (afterDash) {
                const match = afterDash.match(/\((Retain|Expire)\)/); // Match (Retain) or (Expire)
                if (match) {
                    formattedValidationFlag += ` - ${match[0]}`; // Append the matched part
                } else if (intendedAction) {
                    formattedValidationFlag += ` - ${intendedAction}`; // Append the intendedAction if no match
                }
            }
            
            // Build HTML for the row based on columns that exist
            let rowHTML = `
                <td class="dataset-col">${row.DataSet || ''}</td>
                <td class="file-row-col">${row.FileRow || ''}</td>
                <td class="contract-col" title="${row['Contract Number']}">${row['Contract Number'] || ''}</td>
                <td class="vendor-id-col">${row['ERP Vendor ID'] || ''}</td>
                <td class="part-num-col editable" data-field="MfgPartNum">${row['Mfg Part Num'] || ''}</td>
                <td class="uom-col editable" data-field="UOM">${row.UOM || ''}</td>
                <td class="qoe-col editable" data-field="QOE">${row.QOE || ''}</td>`;
                
            // Only add action columns if they exist in the table
            if (hasActionCols) {
                rowHTML += `
                    <td class="action-col">
                        <button class="btn btn-danger btn-sm drop-btn">Drop</button>
                        <button class="btn btn-secondary btn-sm revert-btn">Revert</button>
                    </td>
                    <td class="action-col">
                        <button class="btn btn-primary btn-sm edit-btn">Edit</button>
                        <button class="btn btn-success btn-sm commit-btn" disabled>Commit</button>
                    </td>`;
            }
            
            // Add remaining columns
            rowHTML += `
                <td class="part-num-col editable" data-field="VendorPartNum">${row['Vendor Part Num'] || ''}</td>
                <td class="price-col">${price}</td>
                <td class="item-col">${row.Item || ''}</td>
                <td class="validation-col" title="${validationFlag}">${formattedValidationFlag}</td>
                <td class="date-col">${effectiveDate}</td>
                <td class="date-col">${expirationDate}</td>
                <td class="description-col" title="${description}">${description}</td>`;
                
            tr.innerHTML = rowHTML;

            if (hasActionCols) {
                if (row.isDrop === 1) {
                    tr.classList.add('table-danger');
                    tr.style.textDecoration = 'line-through';
                    tr.querySelector('.drop-btn').innerText = 'Keep';
                    tr.querySelector('.revert-btn').disabled = true;
                    tr.querySelector('.edit-btn').disabled = true;
                    tr.querySelector('.commit-btn').disabled = true;
                }

                const dropBtn = tr.querySelector('.drop-btn');
                if (row.isDrop === 1) {
                    dropBtn.innerText = 'Keep';
                    dropBtn.classList.add('keep-state');
                } else {
                    dropBtn.innerText = 'Drop';
                    dropBtn.classList.add('drop-state');
                }
            }
                        
            errorRows.appendChild(tr);
        });

        // Add event listeners for all buttons
        if (hasActionCols) {
            setupButtonListeners();
        }
    }

    // Function to setup button event listeners
    function setupButtonListeners() {
        // Edit button event
        document.querySelectorAll('.edit-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const row = this.closest('tr');
                toggleEditMode(row, true);
            });
        });

        // Commit button event
        document.querySelectorAll('.commit-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const row = this.closest('tr');
                saveEdits(row);
            });
        });

        // Drop button event
        document.querySelectorAll('.drop-btn').forEach(btn => {
            btn.addEventListener('click', function () {
                const row = this.closest('tr');
                toggleDrop(row);
            });
        });

        // Revert button event
        document.querySelectorAll('.revert-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const row = this.closest('tr');
                revertRow(row);
            });
        });
    }

    // Function to toggle edit mode for a row
    function toggleEditMode(row, enabled) {
        const editableCells = row.querySelectorAll('.editable');
        editableCells.forEach(cell => {
            if (enabled) {
                // Before making editable, clean up the cell content if it has previous edits
                const hasBeenEditedBefore = cell.querySelector('del');
                if (hasBeenEditedBefore) {
                    // Extract just the current value (the last div in the cell)
                    const currentValue = cell.querySelector('div:last-child')?.innerText || cell.innerText;
                    // Reset the cell to just show the current value for editing
                    cell.innerHTML = currentValue;
                    
                    // Store the committed value as the original for this edit session
                    cell.setAttribute('data-original-display', `<div><del>${cell.getAttribute('data-baseline')}</del></div><div>${currentValue}</div>`);
                } else {
                    // For cells that haven't been edited before, store the simple original value
                    cell.setAttribute('data-original-display', cell.innerText.trim());
                }
                
                // Now make the cell editable
                cell.contentEditable = true;
                cell.classList.add('editing');
                
                // Store the baseline value (only if not already set)
                if (!cell.hasAttribute('data-baseline')) {
                    cell.setAttribute('data-baseline', cell.innerText.trim());
                }
                // Store current value for immediate comparison
                cell.setAttribute('data-original', cell.innerText.trim());
            } else {
                cell.contentEditable = false;
                cell.classList.remove('editing');
                // Clean up the temporary display attribute
                cell.removeAttribute('data-original-display');
            }
        });

        // Add or remove input validation
        if (enabled) {
            addInputValidation(row);
            // Add click-outside listener when entering edit mode
            document.addEventListener('click', handleClickOutside);
        } else {
            removeInputValidation(row);
            // Remove click-outside listener when exiting edit mode
            document.removeEventListener('click', handleClickOutside);
        }

        // Toggle commit button
        const commitBtn = row.querySelector('.commit-btn');
        commitBtn.disabled = !enabled;
    }

    // Function to validate and format cell input
    function validateCellInput(cell) {
        const field = cell.getAttribute('data-field');
        let value = cell.innerText;
        
        // Store cursor position before validation
        const selection = window.getSelection();
        const range = selection.getRangeAt(0);
        const cursorOffset = range.startOffset;
        
        let newValue;
        let originalLength = value.length;
        
        if (field === 'QOE') {
            // Allow only digits (0-9) for QOE
            newValue = value.replace(/[^0-9]/g, '');
        } else {
            // Allow only specified characters for text fields
            newValue = value.replace(/[^0-9A-Za-z\/\\,.;:'"~!#$%^&*()\-=+{}[\]\s]/g, '');
            newValue = newValue.toUpperCase(); // Convert to uppercase
        }
        
        // Only update if the value changed
        if (value !== newValue) {
            cell.innerText = newValue;
            
            // Calculate new cursor position
            const lengthDifference = originalLength - newValue.length;
            const newCursorPosition = Math.max(0, cursorOffset - lengthDifference);
            
            // Restore cursor position
            const newRange = document.createRange();
            const textNode = cell.firstChild;
            
            if (textNode && textNode.nodeType === Node.TEXT_NODE) {
                const maxOffset = Math.min(newCursorPosition, textNode.textContent.length);
                newRange.setStart(textNode, maxOffset);
                newRange.setEnd(textNode, maxOffset);
                
                selection.removeAllRanges();
                selection.addRange(newRange);
            }
        }
    }

    // Function to add input validation to editable cells
    function addInputValidation(row) {
        const editableCells = row.querySelectorAll('.editable');
        editableCells.forEach(cell => {
            // Remove any existing event listener to avoid duplicates
            cell.removeEventListener('input', cell.validationHandler);
            
            // Create a bound validation handler
            cell.validationHandler = function() {
                validateCellInput(cell);
            };
            
            // Add the input event listener
            cell.addEventListener('input', cell.validationHandler);
        });
    }

    // Function to remove input validation from editable cells
    function removeInputValidation(row) {
        const editableCells = row.querySelectorAll('.editable');
        editableCells.forEach(cell => {
            if (cell.validationHandler) {
                cell.removeEventListener('input', cell.validationHandler);
                delete cell.validationHandler;
            }
        });
    }

    // Function to handle clicking outside edit mode
    function handleClickOutside(event) {
        // Find the currently editing row
        const editingRow = document.querySelector('tr .editing')?.closest('tr');
        
        if (!editingRow) return; // No row is being edited
        
        // Check if the click was outside the editing row
        if (!editingRow.contains(event.target)) {
            // Revert any changes and exit edit mode
            revertRowChanges(editingRow);
            toggleEditMode(editingRow, false);
        }
    }

    // Function to revert row changes to previous state
    function revertRowChanges(row) {
        const editableCells = row.querySelectorAll('.editable');
        editableCells.forEach(cell => {
            const originalDisplay = cell.getAttribute('data-original-display');
            if (originalDisplay !== null) {
                // Restore the original display (either simple text or crossed-out + new value)
                cell.innerHTML = originalDisplay;
            }
        });
    }

    // Function to save edits
    function saveEdits(row) {
        const pkid = row.getAttribute('data-pkid');
        const dataSet = row.getAttribute('data-dataset');
        const fileRow = row.getAttribute('data-filerow');
        const contractNumber = row.getAttribute('data-contract');
        const mfgPartNumBsl = row.getAttribute('data-mfg-part-num-bsl');
        const vendorPartNumBsl = row.getAttribute('data-vendor-part-num-bsl');
        const uomBsl = row.getAttribute('data-uom-bsl');
        const qoeBsl = row.getAttribute('data-qoe-bsl');

        // Get the current values from the row for all required fields
        // const mfgPartNum = row.querySelector('[data-field="MfgPartNum"]').innerText.trim();
        // const vendorPartNum = row.querySelector('[data-field="VendorPartNum"]').innerText.trim();
        // const uom = row.querySelector('[data-field="UOM"]').innerText.trim();
        // const qoeElement = row.querySelector('[data-field="QOE"]');
        // const qoe = qoeElement ? parseInt(qoeElement.innerText.trim()) || 0 : 0;

        
        const editableCells = row.querySelectorAll('.editable');
        const edits = [];
        let isChanged = false;
        let isWrong = 0;
        
        editableCells.forEach(cell => {
            const field = cell.getAttribute('data-field');
            const baselineValue = cell.getAttribute('data-baseline');
            const originalValue = cell.getAttribute('data-original');
            let newValue = cell.innerText.trim();

            
            // Normalize values
            if (field === 'QOE') {
                newValue = parseInt(newValue) || 0;
            } else {
                newValue = newValue.toUpperCase();
            }
            
            // Check if value was changed compared to baseline
            // as soon as we detect a change, we mark it as changed
            // also if the value is different from last committed value, we should also mark it as changed
            if (baselineValue !== newValue.toString() || originalValue !== newValue.toString()) {
                isChanged = true;

                if (baselineValue !== newValue.toString()) {
                    isWrong = 1; // Mark as wrong if any value changes
                }

                edits.push({
                    field,
                    newValue
                });
            } 

        });
        
        // If nothing changed, just disable edit mode
        if (!isChanged) {
            toggleEditMode(row, false);
            return;
        }
        
        // Prepare data for saving
        const editData = {
            task_id: taskId,
            pkid: pkid,
            data_set: dataSet,
            file_row: fileRow,
            contract_number: contractNumber,
            mfg_part_num: mfgPartNumBsl,
            vendor_part_num: vendorPartNumBsl,
            uom: uomBsl,
            qoe: qoeBsl,
            is_drop: 0,
            is_wrong: isWrong,
            edits: edits
        };
        
        // Send data to server
        fetch(getApiUrl('/data-export/task/errors/edit'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(editData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update UI to reflect changes
                toggleEditMode(row, false);
                
                // Update cells to show crossed-out old value and new value
                editableCells.forEach(cell => {
                    const baselineValue = cell.getAttribute('data-baseline');
                    let newValue = cell.innerText.trim();
                    
                    // Only update display if value changed from baseline
                   if (baselineValue !== newValue.toString()) {
                        // Always show only the original baseline value (crossed out) and the latest value
                        cell.innerHTML = `<div><del>${baselineValue}</del></div><div>${newValue}</div>`;
                        cell.contentEditable = false;
                    } else {
                        // If user has edited back to the original value, remove the strikethrough
                        cell.innerHTML = baselineValue;
                        cell.contentEditable = false;
                    }
                });
                
                // If marked as wrong, highlight the row
                if (isWrong === 1) {
                    row.classList.add('table-warning');
                }

                // if everything is same as baseline, see this as making a revert
                if (isWrong === 0) {
                    // revert the row back to baseline value
                    revertRow(row);
                }
                
                showSuccessMessage('Changes saved successfully');
            } else {
                showErrorMessage(`Error: ${data.message}`);
            }
        })
        .catch(error => {
            showErrorMessage(`Error saving changes: ${error.message}`);
        });
    }


    function toggleDrop(row) {
        const pkid = row.getAttribute('data-pkid');
        const dataSet = row.getAttribute('data-dataset');
        const fileRow = row.getAttribute('data-filerow');
        const contractNumber = row.getAttribute('data-contract');
        const mfgPartNumBsl = row.getAttribute('data-mfg-part-num-bsl');
        const vendorPartNumBsl = row.getAttribute('data-vendor-part-num-bsl');
        const uomBsl = row.getAttribute('data-uom-bsl');
        const qoeBsl = row.getAttribute('data-qoe-bsl');

        // Check the current state of the row (dropped or not)
        const isDropped = row.classList.contains('table-danger');
        
        // If we're about to drop a record and it's in edit mode, exit edit mode first
        if (!isDropped && row.querySelector('.editing')) {
            toggleEditMode(row, false);
        }
        
        // If we're about to drop a record, first revert any existing edits visually
        if (!isDropped) {
            // First revert any visual edits to match what's happening in the backend
            const editableCells = row.querySelectorAll('.editable');
            editableCells.forEach(cell => {
                const baselineValue = cell.getAttribute('data-baseline');
                if (baselineValue) {
                    cell.innerHTML = baselineValue;
                    // Remove any edit-related visual indicators
                    cell.classList.remove('table-warning');
                }
            });
            
            // Remove the warning highlight from the row if it exists
            row.classList.remove('table-warning');
        }

        // Prepare data for toggling
        const toggleData = {
            task_id: taskId,
            pkid: pkid,
            data_set: dataSet,
            file_row: fileRow,
            contract_number: contractNumber,
            mfg_part_num: mfgPartNumBsl,
            vendor_part_num: vendorPartNumBsl,
            uom: uomBsl,
            qoe: qoeBsl,
            is_drop: isDropped ? 0 : 1, // Toggle drop state
            is_wrong: 0,
            field: 'DROP',
            new_value: ''
        };

        // Send data to server
        fetch(getApiUrl('/data-export/task/errors/toggle-drop'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(toggleData),
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Toggle the row's state
                    if (isDropped) {
                        // Reinstate the row
                        row.classList.remove('table-danger');
                        row.querySelectorAll('td').forEach(cell => {
                            if (!cell.classList.contains('action-col')) {
                                cell.style.textDecoration = 'none';
                            }
                        });
                        const dropBtn = row.querySelector('.drop-btn');
                        dropBtn.innerText = 'Drop';
                        dropBtn.classList.remove('keep-state');
                        dropBtn.classList.add('drop-state');
                        
                        // Re-enable buttons
                        const revertBtn = row.querySelector('.revert-btn');
                        const editBtn = row.querySelector('.edit-btn');
                        if (revertBtn) revertBtn.disabled = false;
                        if (editBtn) editBtn.disabled = false;
                        // Commit stays disabled until edit mode is activated
                    } else {
                        // Mark the row as dropped
                        row.classList.add('table-danger');
                        row.querySelectorAll('td').forEach(cell => {
                            if (!cell.classList.contains('action-col')) {
                                cell.style.textDecoration = 'line-through';
                            }
                        });
                        const dropBtn = row.querySelector('.drop-btn');
                        dropBtn.innerText = 'Keep';
                        dropBtn.classList.remove('drop-state');
                        dropBtn.classList.add('keep-state');
                        
                        // Disable buttons
                        const revertBtn = row.querySelector('.revert-btn');
                        const editBtn = row.querySelector('.edit-btn');
                        const commitBtn = row.querySelector('.commit-btn');
                        if (revertBtn) revertBtn.disabled = true;
                        if (editBtn) editBtn.disabled = true;
                        if (commitBtn) commitBtn.disabled = true;
                    }

                    showSuccessMessage(isDropped ? 'Row reinstated successfully' : 'Row marked as dropped');
                } else {
                    showErrorMessage(`Error: ${data.message}`);
                }
            })
            .catch(error => {
                showErrorMessage(`Error toggling drop state: ${error.message}`);
            });
    }


    // this is different from the reverRowChange function, this function will revert the row to its baseline values
    function revertRow(row) {
        const pkid = row.getAttribute('data-pkid');
        const dataSet = row.getAttribute('data-dataset');
        const fileRow = row.getAttribute('data-filerow');
        const contractNumber = row.getAttribute('data-contract');

        // // Confirm before reverting
        // if (!confirm('Are you sure you want to revert all changes for this row? This cannot be undone.')) {
        //     return;
        // }
        
        // Prepare data for reverting
        const revertData = {
            task_id: taskId,
            pkid: pkid,
            data_set: dataSet,
            file_row: fileRow,
            contract_number: contractNumber
        };
        
        // Send data to server
        fetch(getApiUrl('/data-export/task/errors/revert'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(revertData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Reset all editable cells to their baseline values
                const editableCells = row.querySelectorAll('.editable');
                editableCells.forEach(cell => {
                    const baselineValue = cell.getAttribute('data-baseline');
                    if (baselineValue) {
                        cell.innerHTML = baselineValue;
                        // Remove any edit-related attributes
                        cell.removeAttribute('data-original');
                    }
                });
                
                // Remove highlighting
                row.classList.remove('table-warning');
                
                showSuccessMessage('Changes reverted successfully');
            } else {
                showErrorMessage(`Error: ${data.message}`);
            }
        })
        .catch(error => {
            showErrorMessage(`Error reverting changes: ${error.message}`);
        });
    }

    // funtcion to revert all rows visually
    function revertAllRowsUI() {
        const errorRows = document.querySelectorAll('#error-rows tr');
        errorRows.forEach(row => {
            // Reset editable cells to their baseline values
            const editableCells = row.querySelectorAll('.editable');
            editableCells.forEach(cell => {
                const baselineValue = cell.getAttribute('data-baseline');
                if (baselineValue) {
                    cell.innerHTML = baselineValue;
                    cell.removeAttribute('data-original');
                }
            });

            // Remove dropped line formatting
            if (row.classList.contains('table-danger')) {
                row.classList.remove('table-danger');
                row.querySelectorAll('td').forEach(cell => {
                    if (!cell.classList.contains('action-col')) {
                        cell.style.textDecoration = 'none';
                    }
                });

                const dropBtn = row.querySelector('.drop-btn');
                dropBtn.innerText = 'Drop';
                dropBtn.classList.remove('keep-state');
                dropBtn.classList.add('drop-state');
            }

            // Re-enable buttons
            const revertBtn = row.querySelector('.revert-btn');
            const editBtn = row.querySelector('.edit-btn');
            if (revertBtn) revertBtn.disabled = false;
            if (editBtn) editBtn.disabled = false;

            // Remove warning highlights
            row.classList.remove('table-warning');
        });
    }

    // Function to handle "Revert All" action
    function revertAllEdits() {
        if (!confirm('Are you sure you want to revert all changes? This action will reset all edits made to this task, including dropped lines.')) {
            return;
        }

        fetch(getApiUrl(`/data-export/task/${taskId}/revert-all`), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Efficiently reset all rows visually
                    revertAllRowsUI();

                    showSuccessMessage('All changes have been reverted successfully.');
                } else {
                    showErrorMessage(`Error: ${data.message}`);
                }
            })
            .catch(error => {
                showErrorMessage(`Error reverting all changes: ${error.message}`);
            });
    }


    // Function to load sync status data
    function loadSyncStatus() {
        fetch(getApiUrl(`/data-export/task/${taskId}/sync-status`))
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    updateSyncStatusTable(data);
                } else {
                    showSyncStatusError(data.message);
                }
            })
            .catch(error => {
                showSyncStatusError(`Failed to load sync status: ${error.message}`);
            });
    }

    // Function to update the sync status table with data
    function updateSyncStatusTable(data) {
        // Update Max Sync %
        const maxSyncCell = document.getElementById('max-sync');
        if (maxSyncCell) {
            maxSyncCell.innerHTML = formatSyncPercentage(data.maxSync);
        }
        
        // Update CCX sync percentages
        const ccxTpSync = document.getElementById('ccx-tp-sync');
        const ccxChangesSync = document.getElementById('ccx-changes-sync');
        if (ccxTpSync) ccxTpSync.innerHTML = formatSyncPercentage(data.ccxTpSync);
        if (ccxChangesSync) ccxChangesSync.innerHTML = formatSyncPercentage(data.ccxChangesSync);
        
        // Update Infor sync percentages
        const inforTpSync = document.getElementById('infor-tp-sync');
        const inforChangesSync = document.getElementById('infor-changes-sync');
        if (inforTpSync) inforTpSync.innerHTML = formatSyncPercentage(data.inforTpSync);
        if (inforChangesSync) inforChangesSync.innerHTML = formatSyncPercentage(data.inforChangesSync);
    }

    // Function to format sync percentage with color coding
    function formatSyncPercentage(value) {
        if (value === null || value === undefined) return 'N/A';
        
        const percent = parseFloat(value).toFixed(2);
        let colorClass = '';
        
        if (percent >= 100) {
            colorClass = 'sync-high';
        } else if (percent >= 90) {
            colorClass = 'sync-medium';
        } else {
            colorClass = 'sync-low';
        }
        
        return `<span class="${colorClass}">${percent}%</span>`;
    }

    // Function to show sync status error
    function showSyncStatusError(message) {
        const table = document.getElementById('sync-status-table');
        if (table) {
            const container = table.parentElement;
            container.innerHTML = `
                <div class="alert alert-danger">
                    <p>Error loading sync status: ${message}</p>
                </div>
            `;
        }
    }


    // Helper functions for messages
    function showSuccessMessage(message) {
        // Implement a toast or alert system for success messages
        alert(message);
    }
    
    function showErrorMessage(message) {
        // Implement a toast or alert system for error messages
        alert(`Error: ${message}`);
    }
});