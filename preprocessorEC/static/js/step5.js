document.addEventListener('DOMContentLoaded', function() {
    // Change mode description toggler
    const updateActionModeSelect = document.getElementById('update_action_mode');
    const changeModeDescriptions = document.querySelectorAll('.update-mode-description');
    
    updateActionModeSelect.addEventListener('change', function() {
        // Hide all descriptions
        changeModeDescriptions.forEach(desc => desc.classList.remove('active-description'));
        
        // Show the selected one
        const selectedDesc = document.getElementById(`${this.value}_mode_desc`);
        if (selectedDesc) {
            selectedDesc.classList.add('active-description');
        }
    });

    const finalizeChangesBtn = document.getElementById('finalize-changes-btn');
    if (finalizeChangesBtn) {
        finalizeChangesBtn.addEventListener('click', function() {
            finalizeChanges();
        });
    }
});

// taking the layout.html global alert function
function showAlert(type, message, timeout = 5000) {
        window.showGlobalAlert(type, message, timeout);
    }

document.addEventListener('click', function(event) {
    // Check if the click is outside the data change table
    const dataChangeTable = document.getElementById('data-change-table-container');
    const referenceTableSection = document.getElementById('reference-table-section');
    
    // If the table containers exist and the click is outside both tables
    if (dataChangeTable && referenceTableSection) {
        const isClickInsideMainTable = dataChangeTable.contains(event.target);
        const isClickInsideReferenceTable = referenceTableSection.contains(event.target);
        
        // If click is outside both tables, clear selections and highlighting
        if (!isClickInsideMainTable && !isClickInsideReferenceTable) {
            // Remove highlighting from all rows
            document.querySelectorAll('#data-change-tbody tr').forEach(row => {
                row.classList.remove('table-active');
            });
            
            // Clear field difference highlighting
            clearFieldDifferenceHighlighting();
            
            // Hide reference table section
            document.getElementById('reference-table-section').style.display = 'none';
        }
    }
});

document.getElementById('view-changes-btn').addEventListener('click', function() {
    // Show loading spinner
    document.getElementById('loading-spinner').style.display = 'block';
    document.getElementById('results-container').style.display = 'none';
    
    // Get the selected change mode (renamed to update_action_mode)
    const updateActionMode = document.getElementById('update_action_mode').value;
    
    // Make AJAX request to show_changes endpoint
    fetch(getApiUrl('/change-simulation/show-changes'), {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        },
        credentials: 'same-origin',
        body: JSON.stringify({
            update_action_mode: updateActionMode
        })
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading spinner
        document.getElementById('loading-spinner').style.display = 'none';
        
        // Display results container
        document.getElementById('results-container').style.display = 'block';
        
        if (data.success) {
            // Show success message using the alert handler
            showAlert('success', data.message);
            
            // Store data for statistics calculation
            const ccxCreateData = data.result.ccx_create || [];
            const ccxUpdateData = data.result.ccx_update || [];
            const ccxExpireData = data.result.ccx_expire || [];
            const tpCreateData = data.result.tp_create || [];
            const tpMuteData = data.result.tp_mute || [];
            const tpMergedData = data.result.tp_merged || [];
            
            // Calculate counts from actual data length
            const ccxCreate = ccxCreateData.length;
            const ccxUpdate = ccxUpdateData.length;
            const ccxDelete = ccxExpireData.length;
            const tpCreate = tpCreateData.length;
            const tpMute = tpMuteData.length;
            const tpMerged = tpMergedData.length;
            
            // Update the static HTML elements with the calculated values
            document.getElementById('ccx-create-count').textContent = ccxCreate;
            document.getElementById('ccx-update-count').textContent = ccxUpdate;
            document.getElementById('ccx-delete-count').textContent = ccxDelete;
            document.getElementById('tp-create-count').textContent = tpCreate;
            document.getElementById('tp-mute-count').textContent = tpMute;
            document.getElementById('tp-merged-count').textContent = tpMerged;

            // Store data for modal display
            window.changeStatsData = {
                ccxCreate: ccxCreateData,
                ccxUpdate: ccxUpdateData,
                ccxExpire: ccxExpireData,
                tpCreate: tpCreateData,
                tpMute: tpMuteData,
                tpMerged: tpMergedData
            };

            // Attach click handlers to cards
            attachCardClickHandlers();
            
            // Store data change items for table
            allDataChangeItems = data.result.data_change_show || [];
            
            // Store reference items for expire rows
            allReferenceItems = data.result.reference_for_expire_rows || [];
            
            // Display network graphs if available
            if (data.result.original_graph_data && data.result.modified_graph_data) {
                document.getElementById('network-graph-container').style.display = 'block';
                
                // Display data change table
                displayDataChangeTable();

                // Show the finalize actions div now that we have changes to review
                document.querySelector('.finalize-actions').style.display = 'flex';
                
                try {
                    // Parse the graph data from JSON
                    const graphDataOriginal = JSON.parse(data.result.original_graph_data);
                    const graphDataModified = JSON.parse(data.result.modified_graph_data);

                    // config object
                    const plotConfig = {
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
                        displaylogo: false,
                        scrollZoom: true,
                        doubleClick: 'reset'
                    };
                    
                    // Create the left Plotly graph
                    Plotly.newPlot('original-network-graph', graphDataOriginal.data, graphDataOriginal.layout, plotConfig)
                        .then(function() {
                            // Force pan mode after the plot is created
                            Plotly.relayout('original-network-graph', {'dragmode': 'pan'});
                        });
                    
                    // Create the right Plotly graph
                    Plotly.newPlot('modified-network-graph', graphDataModified.data, graphDataModified.layout, plotConfig)
                        .then(function() {
                            // Force pan mode after the plot is created
                            Plotly.relayout('modified-network-graph', {'dragmode': 'pan'});
                        });
                    
                } catch (error) {
                    document.getElementById('network-graph-container').innerHTML = 
                        '<div class="alert alert-warning">Error displaying graphs. Please try again or contact support.</div>';
                }
            } else {
                // Hide graph container if no data
                document.getElementById('network-graph-container').style.display = 'none';
            }
        } else {
            // Show error message using the global alert handler
            if (typeof window.showGlobalAlert === 'function') {
                window.showGlobalAlert('danger', data.message);
            } else {
                console.log('ERROR: ' + data.message);
            }
            
            // Display error message - replace with top-level alert instead of updating div
            document.getElementById('network-graph-container').style.display = 'none';
        }
    })
    .catch(error => {
        // Hide loading spinner
        document.getElementById('loading-spinner').style.display = 'none';
        
        // Show error message using the global alert handler
        if (typeof window.showGlobalAlert === 'function') {
            window.showGlobalAlert('danger', "An error occurred while processing your request. Please try again.");
        } else {
            console.log('ERROR: An error occurred while processing your request.');
        }
        
        // Display error - replace with top-level alert instead of updating div
        document.getElementById('results-container').style.display = 'block';
        document.getElementById('network-graph-container').style.display = 'none';
    });
});

// Function to attach click handlers to change statistics cards
function attachCardClickHandlers() {
    // CCX Create card
    document.getElementById('ccx-create-count').closest('.change-stat-card').addEventListener('click', function() {
        showChangeStatsModal('CCX Create', window.changeStatsData.ccxCreate);
    });
    
    // CCX Update card
    document.getElementById('ccx-update-count').closest('.change-stat-card').addEventListener('click', function() {
        showChangeStatsModal('CCX Update', window.changeStatsData.ccxUpdate);
    });
    
    // CCX Expire card
    document.getElementById('ccx-delete-count').closest('.change-stat-card').addEventListener('click', function() {
        showChangeStatsModal('CCX Expire', window.changeStatsData.ccxExpire);
    });
    
    // TP Create card
    document.getElementById('tp-create-count').closest('.change-stat-card').addEventListener('click', function() {
        showChangeStatsModal('TP Create', window.changeStatsData.tpCreate);
    });
    
    // TP Mute card
    document.getElementById('tp-mute-count').closest('.change-stat-card').addEventListener('click', function() {
        showChangeStatsModal('TP Mute', window.changeStatsData.tpMute);
    });
    
    // TP Merged card
    document.getElementById('tp-merged-count').closest('.change-stat-card').addEventListener('click', function() {
        showChangeStatsModal('TP Merged', window.changeStatsData.tpMerged);
    });
}

// Function to show change statistics modal
function showChangeStatsModal(title, data) {
    // Set modal title
    document.getElementById('changeStatsModalLabel').textContent = `${title} Details`;
    
    const tableBody = document.getElementById('changeStatsTableBody');
    const noDataMessage = document.getElementById('noStatsDataMessage');
    const tableContainer = document.getElementById('changeStatsTableContainer');
    
    // Clear existing content
    tableBody.innerHTML = '';
    
    if (!data || data.length === 0) {
        // Show no data message
        tableContainer.style.display = 'none';
        noDataMessage.style.display = 'block';
    } else {
        // Show table and populate data
        tableContainer.style.display = 'block';
        noDataMessage.style.display = 'none';
        
        data.forEach(item => {
            const row = document.createElement('tr');
            
            // Format contract price
            const contractPrice = item['Contract Price'] ? 
                '$' + parseFloat(item['Contract Price']).toFixed(2) : 'N/A';
            
            // Format actions
            const formattedPrimaryAction = formatActionText(item['Primary Action'] || '');
            const formattedActualAction = formatActionText(item['Actual Action'] || '');
            
            // Truncate description if too long
            const description = item['Description'] || '';
            const displayDescription = description.length > 50 ? 
                description.substring(0, 47) + '...' : description;
            
            row.innerHTML = `
                <td><span class="${getPrimaryActionClass(item['Primary Action'])}">${formattedPrimaryAction}</span></td>
                <td><span class="${getActualActionClass(item['Actual Action'])}">${formattedActualAction}</span></td>
                <td>${item['ERP Vendor ID'] || ''}</td>
                <td>${item['Contract Number'] || ''}</td>
                <td>${item['Mfg Part Num'] || ''}</td>
                <td>${item['Vendor Part Num'] || ''}</td>
                <td>${item['UOM'] || ''}</td>
                <td>${item['QOE'] || ''}</td>
                <td>${contractPrice}</td>
                <td>${item['Effective Date'] || ''}</td>
                <td>${item['Expiration Date'] || ''}</td>
                <td title="${description}">${displayDescription}</td>
                <td>${item['File Row'] || ''}</td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    // Show modal
    $('#changeStatsModal').modal('show');
}

// Variables for data change table
let allDataChangeItems = [];
let allReferenceItems = []; // Add reference data storage
let sortFieldDataChange = null;
let sortDirectionDataChange = 'asc';

let checkboxStates = {};

function getItemCompositeKey(item) {
    return [
        item['Contract Number'],
        item['ERP Vendor ID'],
        item['Mfg Part Num'],
        item['Vendor Part Num'],
        item['UOM'],
        item['File Row']
    ].join('|');
}

// Function to display data change table
function displayDataChangeTable() {
    if (allDataChangeItems.length === 0) {
        document.getElementById('data-change-controls').style.display = 'none';
        document.getElementById('data-change-table-container').style.display = 'none';
        document.getElementById('no-data-message').style.display = 'block';
        return;
    }
    
    // Show controls and table
    document.getElementById('data-change-controls').style.display = 'flex';
    document.getElementById('data-change-table-container').style.display = 'block';
    document.getElementById('no-data-message').style.display = 'none';
    
    // Reset sorting
    sortFieldDataChange = null;
    sortDirectionDataChange = 'asc';
    document.querySelectorAll('#data-change-table th').forEach(header => {
        header.classList.remove('sort-asc', 'sort-desc');
    });
    
    // Render table
    renderDataChangeTable(allDataChangeItems);
    
    // Attach event listeners - remove existing ones first
    attachDataChangeEventListeners(true);
}

// Function to format action text for better display
function formatActionText(action) {
    if (!action) return '';
    
    // Replace "Expire then Create" patterns
    if (action === 'Expire then Create (Create)') {
        return 'ETC (Create)';
    } else if (action === 'Expire then Create (Expire)') {
        return 'ETC (Expire)';
    } else if (action === 'Update (Existing)') {
        return 'Update (Curr.)';
    }
    
    // Return original text for other actions
    return action;
}

// Function to get CSS class for Intended Action
function getIntendedActionClass(action) {
    if (!action) return '';
    const actionLower = action.toLowerCase();
    if (actionLower.includes('expire')) return 'intended-action-expire';
    if (actionLower.includes('upsert')) return 'intended-action-upsert';
    return '';
}

// Function to get CSS class for Primary Action
function getPrimaryActionClass(action) {
    if (!action) return '';
    const actionLower = action.toLowerCase();
    if (actionLower.includes('expire')) return 'primary-action-expire';
    if (actionLower.includes('create')) return 'primary-action-create';
    if (actionLower.includes('update')) return 'primary-action-update';
    if (actionLower.includes('no change')) return 'primary-action-no-change';
    return '';
}

// Function to get CSS class for Actual Action
function getActualActionClass(action) {
    if (!action) return '';
    const actionLower = action.toLowerCase();
    
    // More specific checks first
    if (actionLower === 'expire then create (create)') {
        return 'actual-action-etc-create';
    }
    
    if (actionLower === 'expire then create (expire)') {
        return 'actual-action-etc-expire';
    }
    
    // Handle pattern matching for various formats
    if (actionLower.includes('etc') && actionLower.includes('create')) {
        return 'actual-action-etc-create';
    }
    
    if (actionLower.includes('etc') && actionLower.includes('expire')) {
        return 'actual-action-etc-expire';
    }
    
    // Generic cases - order matters here
    if (actionLower.includes('no change')) return 'actual-action-no-change';
    if (actionLower.includes('update') && actionLower.includes('existing')) return 'actual-action-update-existing';
    if (actionLower.includes('update') && actionLower.includes('new')) return 'actual-action-update-new';
    // These need to be last since they're more general
    if (actionLower.includes('expire')) return 'actual-action-expire';
    if (actionLower.includes('create')) return 'actual-action-create';
    
    return '';
}

// Function to find and mark paired update items
function findUpdatePairs(items) {
    const pairs = {};
    
    // First pass: group items by file row + contract number
    items.forEach(item => {
        const fileRow = item['File Row'];
        const contractNumber = item['Contract Number'];
        
        if (fileRow && contractNumber) {
            const pairKey = `${fileRow}:${contractNumber}`;
            
            if (!pairs[pairKey]) {
                pairs[pairKey] = [];
            }
            
            pairs[pairKey].push(item);
        }
    });
    
    // Second pass: identify pairs and mark them
    Object.values(pairs).forEach(itemGroup => {
        // Only process if we have multiple items in the group
        if (itemGroup.length >= 2) {
            let etcExpire = null;
            let etcCreate = null;
            let updateExisting = null;
            let updateNew = null;
            
            // Identify item types in the group
            itemGroup.forEach(item => {
                const actualAction = item['Actual Action'] || '';
                
                if (actualAction === 'Expire then Create (Expire)' || actualAction === 'ETC (Expire)') {
                    etcExpire = item;
                } else if (actualAction === 'Expire then Create (Create)' || actualAction === 'ETC (Create)') {
                    etcCreate = item;
                } else if (actualAction === 'Update (Existing)' || actualAction === 'Update (Curr.)') {
                    updateExisting = item;
                } else if (actualAction === 'Update (New)') {
                    updateNew = item;
                }
            });
            
            // Mark paired items for highlighting
            if (etcExpire && etcCreate) {
                etcExpire._hasPair = true;
                etcExpire._pairType = 'etc-expire';
                etcCreate._hasPair = true;
                etcCreate._pairType = 'etc-create';
                etcExpire._pairWith = etcCreate;
                etcCreate._pairWith = etcExpire;
            } else if (updateExisting && updateNew) {
                updateExisting._hasPair = true;
                updateExisting._pairType = 'update-existing';
                updateNew._hasPair = true;
                updateNew._pairType = 'update-new';
                updateExisting._pairWith = updateNew;
                updateNew._pairWith = updateExisting;
            }
        }
    });
    
    return items;
}

// Compare fields and apply bold formatting to differences
function highlightPairDifferences(item, pairItem, fieldName, cellValue) {
    // Fields we want to check for differences
    const highlightFields = ['ERP Vendor ID', 'Mfg Part Num', 'Vendor Part Num', 'UOM', 'QOE', 'Contract Price'];
    
    // Only highlight specified fields for create/new items
    if (!highlightFields.includes(fieldName) || 
        (item._pairType !== 'etc-create' && item._pairType !== 'update-new')) {
        return cellValue;
    }
    
    // Get values for comparison
    let itemValue = item[fieldName];
    let pairValue = pairItem[fieldName];
    
    // Special handling for Contract Price (numeric comparison)
    if (fieldName === 'Contract Price') {
        // Extract numeric values for comparison (remove $ and commas)
        const itemPrice = parseFloat(String(itemValue).replace(/[$,]/g, '')) || 0;
        const pairPrice = parseFloat(String(pairValue).replace(/[$,]/g, '')) || 0;
        
        if (Math.abs(itemPrice - pairPrice) > 0.001) {
            return `<strong style="color: #f39c12;">${cellValue}</strong>`;
        }
    } 
    // String comparison for other fields
    else if (String(itemValue || '') !== String(pairValue || '')) {
        return `<strong style="color: #f39c12;">${cellValue}</strong>`;
    }
    
    return cellValue;
}

// Function to render data change table
function renderDataChangeTable(items) {
    const tbody = document.getElementById('data-change-tbody');
    tbody.innerHTML = '';
    
    try {
        // Sort items if needed
        if (sortFieldDataChange) {
            items = [...items].sort((a, b) => {
                let aVal = a[sortFieldDataChange] || '';
                let bVal = b[sortFieldDataChange] || '';
                
                // Add the "Do Not Expire" column sorting logic
                if (sortFieldDataChange === 'Do Not Expire') {
                    // First check if items can have checkboxes
                    const aCanPreventExpire = (a['Intended Action'] === 'Upsert' && a['Primary Action'] === 'Expire CCX') ||
                                            (a['Intended Action'] === 'Expire' && a['Item'] !== '');
                    const bCanPreventExpire = (b['Intended Action'] === 'Upsert' && b['Primary Action'] === 'Expire CCX') ||
                                            (b['Intended Action'] === 'Expire' && b['Item'] !== '');
                    
                    // First sort by whether checkbox exists (can prevent expire)
                    if (aCanPreventExpire !== bCanPreventExpire) {
                        return sortDirectionDataChange === 'asc' ? 
                            (aCanPreventExpire ? -1 : 1) : 
                            (aCanPreventExpire ? 1 : -1);
                    }
                    
                    // Then sort by checkbox state (if both have checkboxes)
                    if (aCanPreventExpire && bCanPreventExpire) {
                        // Get composite keys for both rows
                        const aKey = getItemCompositeKey(a);
                        const bKey = getItemCompositeKey(b);
                        
                        // Check their states in checkboxStates
                        const aChecked = checkboxStates[aKey] || a['Do Not Expire'] === true;
                        const bChecked = checkboxStates[bKey] || b['Do Not Expire'] === true;
                        
                        return sortDirectionDataChange === 'asc' ? 
                            (aChecked ? -1 : 1) : 
                            (aChecked ? 1 : -1);
                    }
                    
                    return 0; // Equal sorting value
                }
                // Handle date fields - convert to timestamps for comparison
                if (['Effective Date', 'Expiration Date'].includes(sortFieldDataChange)) {
                    // Parse dates and handle invalid dates
                    const parseDate = (dateStr) => {
                        if (!dateStr) return 0;
                        const date = new Date(dateStr);
                        return isNaN(date) ? 0 : date.getTime();
                    };
                    
                    aVal = parseDate(aVal);
                    bVal = parseDate(bVal);
                }
                // Handle numeric fields
                else if (['Contract Price', 'QOE'].includes(sortFieldDataChange)) {
                    aVal = parseFloat(aVal) || 0;
                    bVal = parseFloat(bVal) || 0;
                }
                // Handle string fields - convert to lowercase for case-insensitive comparison
                else if (typeof aVal === 'string' && typeof bVal === 'string') {
                    aVal = aVal.toLowerCase();
                    bVal = bVal.toLowerCase();
                }
                
                // Compare based on direction
                if (sortDirectionDataChange === 'asc') {
                    return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
                } else {
                    return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
                }
            });
        }
        
        // Find and mark paired update items
        items = findUpdatePairs(items);

        items.forEach((item, index) => {
            const row = document.createElement('tr');
            
            // Add click handler for row selection
            row.addEventListener('click', function() {
                // Remove highlight from all rows
                document.querySelectorAll('#data-change-tbody tr').forEach(r => {
                    r.classList.remove('table-active');
                });
                
                // Highlight selected row
                this.classList.add('table-active');
                
                // Show reference table for this row
                showReferenceForRow(item);
            });
            
            // Add cursor pointer style
            row.style.cursor = 'pointer';
            
            // Format contract price
            const contractPrice = item['Contract Price'] ? 
                '$' + parseFloat(item['Contract Price']).toFixed(2) : 'N/A';
            
            // Check if this row can have "Do Not Expire" checkbox
            const canPreventExpire = 
                // Case 1: For Upsert intended actions, keep using Primary Action check
                (item['Intended Action'] === 'Upsert' && item['Primary Action'] === 'Expire CCX') || 
                // Case 2: For Expire intended actions, use Item field not empty check
                (item['Intended Action'] === 'Expire' && item['Item'] !== '');

            // Use composite key instead of just file row
            const compositeKey = getItemCompositeKey(item);
            let isChecked = false;
            
            // Check our state tracker first using composite key
            if (compositeKey in checkboxStates) {
                isChecked = checkboxStates[compositeKey];
            } else {
                // If not in our state tracker yet, use the 'Do Not Expire' property from data
                isChecked = item['Do Not Expire'] === true;
                // Initialize our state tracker with composite key
                checkboxStates[compositeKey] = isChecked;
            }
            
            // Create checkbox cell with correct checked state and store the composite key
            const checkboxHtml = canPreventExpire ? 
                `<input type="checkbox" class="expire-checkbox" data-index="${index}" data-key="${compositeKey}" ${isChecked ? 'checked' : ''}>` : 
                '<span class="text-muted">N/A</span>';
            
            // Format actions for better visibility
            const formattedPrimaryAction = formatActionText(item['Primary Action'] || '');
            const formattedActualAction = formatActionText(item['Actual Action'] || '');
            
            // Get CSS classes for color coding
            const intendedActionClass = getIntendedActionClass(item['Intended Action']);
            const primaryActionClass = getPrimaryActionClass(item['Primary Action']);
            const actualActionClass = getActualActionClass(item['Actual Action']);
            
            // Get description from Description field and limit to 40 chars
            const description = item['Description'] || '';
            const displayDescription = description.length > 40 ? 
                description.substring(0, 37) + '...' : description;

            // helper function to format pair-based highlighting
            const getHighlightedCell = (fieldName, content) => {
                if (item._hasPair && item._pairWith) {
                    return highlightPairDifferences(item, item._pairWith, fieldName, content);
                }
                return content;
            };
            
            // Rearranged columns according to new order with color classes wrapped in spans
            row.innerHTML = `
                <td><span class="${intendedActionClass}">${item['Intended Action'] || ''}</span></td>
                <td><span class="${primaryActionClass}">${formattedPrimaryAction}</span></td>
                <td><span class="${actualActionClass}">${formattedActualAction}</span></td>
                <td class="checkbox-col">${checkboxHtml}</td>
                <td class="item-col">${item['Item'] || ''}</td>
                <td class="erp-vendor-id-col">${getHighlightedCell('ERP Vendor ID', item['ERP Vendor ID'] || '')}</td>
                <td class="contract-col">${item['Contract Number'] || ''}</td>
                <td class="part-num-col">${getHighlightedCell('Mfg Part Num', item['Mfg Part Num'] || '')}</td>
                <td class="part-num-col">${getHighlightedCell('Vendor Part Num', item['Vendor Part Num'] || '')}</td>
                <td>${getHighlightedCell('UOM', item['UOM'] || '')}</td>
                <td>${getHighlightedCell('QOE', item['QOE'] || '')}</td>
                <td>${getHighlightedCell('Contract Price', contractPrice)}</td>
                <td class="date-col">${item['Effective Date'] || ''}</td>
                <td class="date-col">${item['Expiration Date'] || ''}</td>
                <td class="description-col" title="${description}">${displayDescription}</td>
            `;
            
            tbody.appendChild(row);
        });

        // Apply current filter after rendering
        filterDataChangeTable();

        // Add event listeners to checkboxes to update our state tracker using composite key
        document.querySelectorAll('#data-change-tbody .expire-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', function() {
                const compositeKey = this.dataset.key;
                if (compositeKey) {
                    checkboxStates[compositeKey] = this.checked;
                }
            });
        });
        
    } catch (err) {
        console.error("Error rendering data change table:", err);
        tbody.innerHTML = '<tr><td colspan="13" class="text-center">Error rendering table data</td></tr>';
    }
}

// Add new function to apply current filter without re-rendering
function applyCurrentFilter() {
    const searchTerm = document.getElementById('data-change-search').value.toLowerCase();
    const searchType = document.getElementById('data-change-search-type').value;
    
    // Only apply filter if there's a search term
    if (searchTerm.trim() === '') {
        // Show all rows if no search term
        document.querySelectorAll('#data-change-tbody tr').forEach(row => {
            row.style.display = '';
        });
        return;
    }
    
    const tableRows = document.querySelectorAll('#data-change-tbody tr');
    let visibleCount = 0;
    
    tableRows.forEach(row => {
        let match = false;
        
        if (searchType === 'contains' || searchType === 'not-contains') {
            // Search in all cells
            const rowText = row.textContent.toLowerCase();
            match = rowText.includes(searchTerm);
            
            // Invert match for not-contains
            if (searchType === 'not-contains') {
                match = !match;
            }
        } else if (searchType === 'contract-only') {
            // Search only in Contract Number column
            const contractCell = row.cells[4];
            if (contractCell) {
                const contractText = contractCell.textContent.toLowerCase();
                match = contractText.includes(searchTerm);
            }
        } else if (searchType === 'action-only') {
            // Search in Action columns
            const primaryActionCell = row.cells[0];
            const actualActionCell = row.cells[1];
            
            if (primaryActionCell && actualActionCell) {
                const primaryActionText = primaryActionCell.textContent.toLowerCase();
                const actualActionText = actualActionCell.textContent.toLowerCase();
                match = primaryActionText.includes(searchTerm) || actualActionText.includes(searchTerm);
            }
        }
        
        // Show/hide row based on match
        row.style.display = match ? '' : 'none';
        if (match) visibleCount++;
    });
    
    // Show/hide no data message based on visible rows
    document.getElementById('no-data-message').style.display = visibleCount > 0 ? 'none' : 'block';
}

// Update the sort click handler to not call attachDataChangeEventListeners
document.querySelectorAll('#data-change-table th[data-sort]').forEach(th => {
    th.addEventListener('click', function() {
        const field = this.dataset.sort;
        
        if (field === sortFieldDataChange) {
            sortDirectionDataChange = sortDirectionDataChange === 'asc' ? 'desc' : 'asc';
        } else {
            sortFieldDataChange = field;
            sortDirectionDataChange = 'asc';
        }
        
        // Update sort indicators
        document.querySelectorAll('#data-change-table th').forEach(header => {
            header.classList.remove('sort-asc', 'sort-desc');
        });
        this.classList.add(sortDirectionDataChange === 'asc' ? 'sort-asc' : 'sort-desc');
        
        // Re-render table with new sort (filter will be applied automatically)
        renderDataChangeTable(allDataChangeItems);
    });
});

// Function to attach event listeners for data change table
// Added parameter to optionally remove existing listeners first
function attachDataChangeEventListeners(removeExisting = false) {
    // Sort listeners - remove and reattach to prevent duplicates
    const sortHeaders = document.querySelectorAll('#data-change-table th[data-sort]');
    
    sortHeaders.forEach(th => {
        if (removeExisting) {
            // Clone and replace to remove all existing event listeners
            const newTh = th.cloneNode(true);
            th.parentNode.replaceChild(newTh, th);
        }
    });
    
    // Re-query the DOM after cloning to get fresh references
    document.querySelectorAll('#data-change-table th[data-sort]').forEach(th => {
        // Add fresh event listener
        th.addEventListener('click', function() {
            const field = this.dataset.sort;
            
            if (field === sortFieldDataChange) {
                sortDirectionDataChange = sortDirectionDataChange === 'asc' ? 'desc' : 'asc';
            } else {
                sortFieldDataChange = field;
                sortDirectionDataChange = 'asc';
            }
            
            // Update sort indicators
            document.querySelectorAll('#data-change-table th').forEach(header => {
                header.classList.remove('sort-asc', 'sort-desc');
            });
            this.classList.add(sortDirectionDataChange === 'asc' ? 'sort-asc' : 'sort-desc');
            
            // Re-render table with new sort
            renderDataChangeTable(allDataChangeItems);
            
            // Don't call attachDataChangeEventListeners here to avoid recursion
        });
    });
    
    // Search functionality
    const searchInput = document.getElementById('data-change-search');
    const searchType = document.getElementById('data-change-search-type');
    const clearSearch = document.getElementById('data-change-clear-search');
    
    if (searchInput && searchType && clearSearch) {
        // Remove existing listeners if they exist
        if (removeExisting) {
            searchInput.removeEventListener('input', filterDataChangeTable);
            searchType.removeEventListener('change', filterDataChangeTable);
            clearSearch.removeEventListener('click', clearSearchHandler);
        }
        
        searchInput.addEventListener('input', filterDataChangeTable);
        searchType.addEventListener('change', filterDataChangeTable);
        clearSearch.addEventListener('click', clearSearchHandler);
    }
    
    // Batch selection buttons
    const selectAllBtn = document.getElementById('select-all-expire-btn');
    const deselectAllBtn = document.getElementById('deselect-all-expire-btn');
    const saveBtn = document.getElementById('save-expire-selections-btn');
    
    if (selectAllBtn && deselectAllBtn && saveBtn) {
        if (removeExisting) {
            selectAllBtn.removeEventListener('click', selectAllHandler);
            deselectAllBtn.removeEventListener('click', deselectAllHandler);
            saveBtn.removeEventListener('click', saveExpireSelections);
        }
        
        selectAllBtn.addEventListener('click', selectAllHandler);
        deselectAllBtn.addEventListener('click', deselectAllHandler);
        saveBtn.addEventListener('click', saveExpireSelections);
    }
}

// Create named handler functions to avoid issues with removeEventListener
function clearSearchHandler() {
    document.getElementById('data-change-search').value = '';
    filterDataChangeTable();
}

function selectAllHandler() {
    // Get all visible rows and update their checkbox states
    document.querySelectorAll('#data-change-tbody tr:not([style*="display: none"]) .expire-checkbox').forEach(checkbox => {
        checkbox.checked = true;
        const compositeKey = checkbox.dataset.key;
        if (compositeKey) {
            checkboxStates[compositeKey] = true;
        }
    });
}

function deselectAllHandler() {
    // Get all visible rows and update their checkbox states
    document.querySelectorAll('#data-change-tbody tr:not([style*="display: none"]) .expire-checkbox').forEach(checkbox => {
        checkbox.checked = false;
        const compositeKey = checkbox.dataset.key;
        if (compositeKey) {
            checkboxStates[compositeKey] = false;
        }
    });
}

// Function to filter data change table - Updated to match step4.html
function filterDataChangeTable() {
    const searchTerm = document.getElementById('data-change-search').value.toLowerCase();
    const searchType = document.getElementById('data-change-search-type').value;
    const tableRows = document.querySelectorAll('#data-change-tbody tr');
    
    let visibleCount = 0;
    
    tableRows.forEach(row => {
        let match = false;
        
        if (searchType === 'contains' || searchType === 'not-contains') {
            // Search in all cells
            const rowText = row.textContent.toLowerCase();
            match = rowText.includes(searchTerm);
            
            // Invert match for not-contains
            if (searchType === 'not-contains') {
                match = !match;
            }
        } else if (searchType === 'contract-only') {
            // Search only in Contract Number column (7th column in the new structure)
            const contractCell = row.cells[6];
            if (contractCell) {
                const contractText = contractCell.textContent.toLowerCase();
                match = contractText.includes(searchTerm);
            }
        } else if (searchType === 'action-only') {
            // Search in Action columns (1st, 2nd, and 3rd columns in the new structure)
            const IntendedActionCell = row.cells[0]; // Intended Action
            const primaryActionCell = row.cells[1]; // Primary Action
            const actualActionCell = row.cells[2]; // Actual Action
            
            if (primaryActionCell && actualActionCell) {
                const primaryActionText = primaryActionCell.textContent.toLowerCase();
                const actualActionText = actualActionCell.textContent.toLowerCase();
                const intendedActionText = IntendedActionCell.textContent.toLowerCase();
                match = primaryActionText.includes(searchTerm) || 
                actualActionText.includes(searchTerm) ||
                intendedActionText.includes(searchTerm);
            }
        } else if (searchType === 'item-only') {
            // Search only in Item column (5th column in the new structure)
            const itemCell = row.cells[4];
            if (itemCell) {
                const itemText = itemCell.textContent.toLowerCase();
                match = itemText.includes(searchTerm);
            }
        }
        
        // Show/hide row based on match
        row.style.display = match ? '' : 'none';
        if (match) visibleCount++;
    });
    
    // Show/hide no data message based on visible rows
    document.getElementById('no-data-message').style.display = visibleCount > 0 ? 'none' : 'block';
}

// Function to save expire selections
function saveExpireSelections() {
    // Show loading spinner
    document.getElementById('loading-spinner').style.display = 'block';
    
    // First, update checkboxStates with current DOM checkbox states
    document.querySelectorAll('#data-change-tbody .expire-checkbox').forEach(checkbox => {
        const compositeKey = checkbox.dataset.key;
        if (compositeKey) {
            checkboxStates[compositeKey] = checkbox.checked;
        }
    });
    
    // Collect checkbox states using composite keys from our state tracker
    const expireSelections = [];
    allDataChangeItems.forEach(item => {
        if (item['Primary Action'] === 'Expire CCX') {
            const compositeKey = getItemCompositeKey(item);
            // Include all items with checkbox states, not just those currently visible
            expireSelections.push({
                contract_number: item['Contract Number'],
                erp_vendor_id: item['ERP Vendor ID'],
                mfg_part_num: item['Mfg Part Num'],
                vendor_part_num: item['Vendor Part Num'],
                uom: item['UOM'],
                primary_action: item['Primary Action'],
                file_row: item['File Row'],
                do_not_expire: checkboxStates[compositeKey] || false
            });
        }
    });
    
    // Send to backend
    fetch(getApiUrl('/change-simulation/update-expire-selections'), {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            expire_selections: expireSelections
        })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('loading-spinner').style.display = 'none';
        
        if (data.success) {
            // Update local data with server response
            if (data.updated_data) {
                // Store the current checkbox states before updating
                const savedCheckboxStates = {...checkboxStates};
                
                // Update the data
                allDataChangeItems = data.updated_data;
                
                // Preserve ALL checkbox states, not just visible ones
                allDataChangeItems.forEach(item => {
                    if (item['Primary Action'] === 'Expire CCX') {
                        const compositeKey = getItemCompositeKey(item);
                        if (compositeKey in savedCheckboxStates) {
                            // Preserve user selections
                            checkboxStates[compositeKey] = savedCheckboxStates[compositeKey];
                            // Also update the item's Do Not Expire property for consistency
                            item['Do Not Expire'] = savedCheckboxStates[compositeKey];
                        } else {
                            // If this is a new key, initialize it from the item
                            checkboxStates[compositeKey] = item['Do Not Expire'] === true;
                        }
                    }
                });
                
                // Re-render the table with updated data and preserved selections
                renderDataChangeTable(allDataChangeItems);
                
                // Reapply the current filter to maintain the filtered view
                filterDataChangeTable();
            }
            
            // Update CCX expire card with new data
            if (data.ccx_expire) {
                // Update the stored data for modal display
                window.changeStatsData.ccxExpire = data.ccx_expire;
                
                // Update the count on the card
                document.getElementById('ccx-delete-count').textContent = data.ccx_expire.length;
            }
            
            // Show alert message
            alert('Expire selections saved successfully!');
        } else {
            // Show error alert
            alert('Error saving expire selections: ' + data.message);
        }
    })
    .catch(error => {
        document.getElementById('loading-spinner').style.display = 'none';
        alert('Error: ' + error.message);
        console.error('Error saving expire selections:', error);
    });
}

// Manual collapse toggle for Action Reference Guide
document.addEventListener('DOMContentLoaded', function() {
    const actionGuideHeader = document.getElementById('action-guide-header');
    const actionGuide = document.getElementById('action-guide');
    
    if (actionGuideHeader && actionGuide) {
        // Initialize - ensure it's collapsed by default
        if (actionGuide.classList.contains('show')) {
            actionGuide.classList.remove('show');
        }
        
        actionGuideHeader.addEventListener('click', function() {
            // Toggle the collapse
            if (actionGuide.classList.contains('show')) {
                actionGuide.classList.remove('show');
                const headerText = actionGuideHeader.querySelector('small');
                if (headerText) {
                    headerText.textContent = 'Click to expand';
                }
            } else {
                actionGuide.classList.add('show');
                const headerText = actionGuideHeader.querySelector('small');
                if (headerText) {
                    headerText.textContent = 'Click to collapse';
                }
            }
        });
    }
});

// Function to show reference data for selected row
function showReferenceForRow(selectedItem) {
    // If selectedItem is null/undefined, hide reference section
    if (!selectedItem || selectedItem['Primary Action'] !== 'Expire CCX') {
        document.getElementById('reference-table-section').style.display = 'none';
        clearFieldDifferenceHighlighting();
        return;
    }

    const referenceSection = document.getElementById('reference-table-section');
    const referenceContainer = document.getElementById('reference-table-container');
    const noReferenceMessage = document.getElementById('no-reference-message');
    const referenceTbody = document.getElementById('reference-tbody');
    
    // Show the reference section
    referenceSection.style.display = 'block';
    
    // Find matching reference record by File Row (corrected field name)
    const fileRow = selectedItem['File Row'];
    let matchingReferences = [];
    
    if (fileRow && allReferenceItems.length > 0) {
        matchingReferences = allReferenceItems.filter(ref => ref['File Row'] === fileRow);
    }
    
    if (matchingReferences.length > 0) {
        // Show reference table and hide no-data message
        referenceContainer.style.display = 'block';
        noReferenceMessage.style.display = 'none';
        
        // Clear any existing highlighting in the main table
        clearFieldDifferenceHighlighting();
        
        // Highlight differences in the selected row
        highlightFieldDifferences(selectedItem, matchingReferences[0]);
        
        // Populate reference table
        referenceTbody.innerHTML = '';
        matchingReferences.forEach(matchingReference => {
            const row = document.createElement('tr');
            
            // Format contract price
            const contractPrice = matchingReference['Contract Price'] ? 
                '$' + parseFloat(matchingReference['Contract Price']).toFixed(2) : 'N/A';
            
            // Format actions for better visibility
            const formattedPrimaryAction = formatActionText(matchingReference['Primary Action'] || '');
            const formattedActualAction = formatActionText(matchingReference['Actual Action'] || '');
            
            // Get CSS classes for color coding
            const intendedActionClass = getIntendedActionClass(matchingReference['Intended Action']);
            const primaryActionClass = getPrimaryActionClass(matchingReference['Primary Action']);
            const actualActionClass = getActualActionClass(matchingReference['Actual Action']);
            
            // Get description and limit to 60 chars
            const description = matchingReference['Description'] || '';
            const displayDescription = description.length > 60 ? 
                description.substring(0, 60) + '...' : description;
            
            row.innerHTML = `
                <td><span class="${intendedActionClass}">${matchingReference['Intended Action'] || ''}</span></td>
                <td><span class="${primaryActionClass}">${formattedPrimaryAction}</span></td>
                <td><span class="${actualActionClass}">${formattedActualAction}</span></td>
                <td class="checkbox-col"><span class="text-muted">N/A</span></td>
                <td class="item-col">${matchingReference['Item'] || ''}</td>
                <td class="erp-vendor-id-col">${matchingReference['ERP Vendor ID'] || ''}</td>
                <td class="contract-col">${matchingReference['Contract Number'] || ''}</td>
                <td class="part-num-col">${matchingReference['Mfg Part Num'] || ''}</td>
                <td class="part-num-col">${matchingReference['Vendor Part Num'] || ''}</td>
                <td class="uom-col">${matchingReference['UOM'] || ''}</td>
                <td class="qoe-col">${matchingReference['QOE'] || ''}</td>
                <td class="price-col">${contractPrice}</td>
                <td class="date-col">${matchingReference['Effective Date'] || ''}</td>
                <td class="date-col">${matchingReference['Expiration Date'] || ''}</td>
                <td class="description-col" title="${description}">${displayDescription}</td>
            `;
            
            referenceTbody.appendChild(row);
        });
    } else {
        // Hide reference table and show no-data message
        referenceContainer.style.display = 'none';
        noReferenceMessage.style.display = 'block';
        
        // Clear any existing highlighting since there's no reference
        clearFieldDifferenceHighlighting();
    }
}

// Function to clear field difference highlighting
function clearFieldDifferenceHighlighting() {
    // Remove highlighting from all cells in the main table
    document.querySelectorAll('#data-change-tbody td').forEach(cell => {
        // Remove field-difference class from cell content
        const content = cell.innerHTML;
        if (content.includes('field-difference')) {
            // Extract the text content and restore it without the highlighting span
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = content;
            const highlightedSpan = tempDiv.querySelector('.field-difference');
            if (highlightedSpan) {
                cell.innerHTML = content.replace(/<span class="field-difference[^>]*>([^<]*)<\/span>/g, '$1');
            }
        }
    });
}

// Function to highlight field differences between selected item and reference
function highlightFieldDifferences(selectedItem, referenceItem) {
    // Find the selected row in the table
    const selectedRow = document.querySelector('#data-change-tbody tr.table-active');
    if (!selectedRow) return;
    
    // Fields to compare (excluding action fields and dates)
    const fieldsToCompare = [
        { field: 'Item', cellIndex: 4 },
        { field: 'ERP Vendor ID', cellIndex: 5 },
        { field: 'Contract Number', cellIndex: 6 },
        { field: 'Mfg Part Num', cellIndex: 7 },
        { field: 'Vendor Part Num', cellIndex: 8 },
        { field: 'UOM', cellIndex: 9 },
        { field: 'QOE', cellIndex: 10 },
        { field: 'Contract Price', cellIndex: 11 }
    ];
    
    fieldsToCompare.forEach(({ field, cellIndex }) => {
        const selectedValue = selectedItem[field] || '';
        const referenceValue = referenceItem[field] || '';
        
        // Compare values (handle Contract Price specially for formatting)
        let isDifferent = false;
        
        if (field === 'Contract Price') {
            // Compare numeric values for contract price
            const selectedPrice = parseFloat(selectedValue) || 0;
            const referencePrice = parseFloat(referenceValue) || 0;
            isDifferent = Math.abs(selectedPrice - referencePrice) > 0.001; // Small tolerance for floating point
        } else {
            // String comparison for other fields
            isDifferent = selectedValue.toString().trim() !== referenceValue.toString().trim();
        }
        
        if (isDifferent) {
            const cell = selectedRow.cells[cellIndex];
            if (cell) {
                // Wrap the cell content in a highlighting span
                const currentContent = cell.innerHTML;
                
                // Check if it's already highlighted to avoid double-wrapping
                if (!currentContent.includes('field-difference')) {
                    cell.innerHTML = `<span class="field-difference">${currentContent}</span>`;
                }
            }
        }
    });
}

function finalizeChanges() {
    // Show loading spinner
    document.getElementById('loading-spinner').style.display = 'block';

    // Show the final expiration checks placeholder
    document.getElementById('final-expiration-checks').style.display = 'block';
    
    fetch(getApiUrl('/change-simulation/finalize-changes'), {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading spinner
        document.getElementById('loading-spinner').style.display = 'none';
        
        if (data.success) {
            // Store the validation data for modal display
            window.validationFlagData = {
                error: data.result.final_errors || [],
                warning: data.result.final_warnings || [],
                check: data.result.final_checks || []
            };

            // Display validation table with final commit data
            if (data.result.final_commit_df) {
                // Reset sort settings for initial display
                sortFieldValidation = null;
                sortDirectionValidation = 'asc';
                
                // Store the data for later sorting without re-fetching
                window.validationData = data.result.final_commit_df;
                
                displayValidationTable(data.result.final_commit_df);
            }

            // Show and populate line operations table if available
            if (data.result.line_operations && data.result.line_operations.length > 0) {
                const lineOperationsContainer = document.getElementById('line-operations-container');
                const lineOperationsTbody = document.getElementById('line-operations-tbody');
                const noLineOperations = document.getElementById('no-line-operations');
                
                // Clear any existing rows
                lineOperationsTbody.innerHTML = '';
                
                // Populate with new data
                data.result.line_operations.forEach(item => {
                    const row = document.createElement('tr');
                    
                    // Get the counts (no change calculation needed)
                    const originalCount = parseInt(item['Total Contract Line Count'] || 0);
                    const finalCount = parseInt(item['Total Contract Line Count (Change Applied)'] || 0);
                    
                    row.innerHTML = `
                        <td>${item['Contract Number'] || ''}</td>
                        <td>${originalCount}</td>
                        <td>${finalCount}</td>
                    `;
                    
                    lineOperationsTbody.appendChild(row);
                });
                
                // Show table, hide empty message
                lineOperationsContainer.style.display = 'block';
                noLineOperations.style.display = 'none';
                } else {
                // If no data, show message
                const lineOperationsContainer = document.getElementById('line-operations-container');
                const noLineOperations = document.getElementById('no-line-operations');
                
                if (lineOperationsContainer && noLineOperations) {
                    lineOperationsContainer.style.display = 'block';
                    document.getElementById('line-operations-tbody').innerHTML = '';
                    noLineOperations.style.display = 'block';
                }
            }
            
            // All checks passed, show success message
            alert('Changes finalized successfully!');

            // Show the commit button first
            document.getElementById('commit-changes-container').style.display = 'inline-block';
            
            // Show the complete step button
            document.getElementById('complete-step-container').style.display = 'inline-block';
            
            // Display the network graphs if available
            if (data.result.modified_graph_data && data.result.r2_graph_data) {
                // Show the existing graph container
                const networkGraphsDiv = document.getElementById('finalized-network-graphs');
                networkGraphsDiv.style.display = 'block';
                
                try {
                    // Parse the graph data
                    const modifiedGraphData = JSON.parse(data.result.modified_graph_data);
                    const r2GraphData = JSON.parse(data.result.r2_graph_data);
                    
                    // Plot config
                    const plotConfig = {
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
                        displaylogo: false,
                        scrollZoom: true,
                        doubleClick: 'reset'
                    };
                    
                    // Create the plots using the existing div elements
                    Plotly.newPlot('finalized-modified-graph', modifiedGraphData.data, modifiedGraphData.layout, plotConfig)
                        .then(() => {
                            Plotly.relayout('finalized-modified-graph', {'dragmode': 'pan'});
                        });
                    
                    Plotly.newPlot('finalized-r2-graph', r2GraphData.data, r2GraphData.layout, plotConfig)
                        .then(() => {
                            Plotly.relayout('finalized-r2-graph', {'dragmode': 'pan'});
                        });
                } catch (error) {
                    console.error('Error creating graphs:', error);
                    networkGraphsDiv.innerHTML = `
                        <div class="alert alert-danger">
                            Error displaying network graphs. Please try again.
                        </div>
                    `;
                }
            }
        } else {
            // Show details about the failed checks
            let message = 'Safety checks failed. Please review the following issues:\n';
            if (data.issues && data.issues.length > 0) {
                data.issues.forEach(issue => {
                    message += `\n- ${issue}`;
                });
            } else {
                message += '\n- ' + (data.message || 'Unknown error occurred');
            }
            
            alert(message);
        }
    })
    .catch(error => {
        // Hide loading spinner
        document.getElementById('loading-spinner').style.display = 'none';
        alert('Error during finalization: ' + error.message);
        console.error('Error finalizing changes:', error);
    });
}


// Function to get CSS class for Validation Flag
function getValidationFlagClass(flag) {
    if (!flag) return 'validation-flag-success';
    
    const flagPrefix = flag.split(' - ')[0]; // Get the part before the dash
    
    if (flagPrefix === 'ERROR') return 'validation-flag-error';
    if (flagPrefix === 'CHECK') return 'validation-flag-check';
    if (flagPrefix === 'WARNING') return 'validation-flag-warning';
    return 'validation-flag-success'; // Default for PASS and others
}


let sortFieldValidation = null;
let sortDirectionValidation = 'asc';
// Function to display validation table with final commit data
function displayValidationTable(validationData) {
    // Get the validation table container and table body
    const validationTableContainer = document.getElementById('validation-table-container');
    const validationTableControls = document.getElementById('validation-table-controls');
    const validationTableBody = document.getElementById('validation-table-body');
    
    if (!validationData || validationData.length === 0) {
        // If no data, don't display the table
        if (validationTableContainer) {
            validationTableContainer.style.display = 'none';
        }
        if (validationTableControls) {
            validationTableControls.style.display = 'none';
        }
        return;
    }

    // Update validation flag counts and show cards
    updateValidationFlagCounts(validationData);

    // show validation table controls
    if (validationTableControls) {
        validationTableControls.style.display = 'flex';
    }
    
    // Clear existing table body content
    validationTableBody.innerHTML = '';
    
    // Apply sorting if needed
    if (sortFieldValidation) {
        validationData = [...validationData].sort((a, b) => {
            let aVal = a[sortFieldValidation] || '';
            let bVal = b[sortFieldValidation] || '';
            
            // Handle date fields
            if (['Effective Date', 'Expiration Date'].includes(sortFieldValidation)) {
                const parseDate = (dateStr) => {
                    if (!dateStr) return 0;
                    const date = new Date(dateStr);
                    return isNaN(date) ? 0 : date.getTime();
                };
                
                aVal = parseDate(aVal);
                bVal = parseDate(bVal);
            }
            // Handle numeric fields
            else if (['Contract Price', 'QOE'].includes(sortFieldValidation)) {
                aVal = parseFloat(aVal) || 0;
                bVal = parseFloat(bVal) || 0;
            }
            // Handle string fields
            else if (typeof aVal === 'string' && typeof bVal === 'string') {
                aVal = aVal.toLowerCase();
                bVal = bVal.toLowerCase();
            }
            
            // Compare based on direction
            if (sortDirectionValidation === 'asc') {
                return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
            } else {
                return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
            }
        });
    }
    
    // Add a row for each validation item
    validationData.forEach(item => {
        const row = document.createElement('tr');
        
        // Format contract price
        const contractPrice = item['Contract Price'] ? 
            '$' + parseFloat(item['Contract Price']).toFixed(2) : '';
        
        // Format description for display (limit length)
        const description = item['Description'] || '';
        const shortDescription = description.length > 40 ? 
            description.substring(0, 37) + '...' : description;

        // Format actions for better visibility
        const formattedPrimaryAction = formatActionText(item['Primary Action'] || '');
        const formattedActualAction = formatActionText(item['Actual Action'] || '');
        
        // Get CSS classes for color coding
        const intendedActionClass = getIntendedActionClass(item['Intended Action']);
        const primaryActionClass = getPrimaryActionClass(item['Primary Action']);
        const actualActionClass = getActualActionClass(item['Actual Action']);
        
        // Get the validation flag and split it for display
        const validationFlag = item['Validation Flag'] || '';
        const flagParts = validationFlag.split(' - ');
        const shortFlag = flagParts[0];
        const flagClass = getValidationFlagClass(validationFlag);
        
        // Create cells in the correct order with proper styling
        const cellsData = [
            // 1. Intended Action
            { content: `<span class="${intendedActionClass}">${item['Intended Action'] || ''}</span>` },
            
            // 2. Primary Action
            { content: `<span class="${primaryActionClass}">${formattedPrimaryAction}</span>` },
            
            // 3. Actual Action
            { content: `<span class="${actualActionClass}">${formattedActualAction}</span>` },
            
            // 4. Validation Flag
            { content: `<span class="${flagClass}" title="${validationFlag}">${shortFlag}</span>` },
            
            // Remaining columns
            { content: item['Item'] || '', className: 'item-col' },
            { content: item['ERP Vendor ID'] || '', className: 'erp-vendor-id-col' },
            { content: item['Contract Number'] || '', className: 'contract-col' },
            { content: item['Mfg Part Num'] || '', className: 'part-num-col' },
            { content: item['Vendor Part Num'] || '', className: 'part-num-col' },
            { content: item['UOM'] || '', className: 'uom-col' },
            { content: item['QOE'] || '', className: 'qoe-col' },
            { content: contractPrice, className: 'price-col' },
            { content: item['Effective Date'] || '', className: 'date-col' },
            { content: item['Expiration Date'] || '', className: 'date-col' },
            { content: shortDescription, className: 'description-col', title: description }
        ];
        
        // Create each cell and add to row
        cellsData.forEach(cellData => {
            const cell = document.createElement('td');
            if (cellData.className) {
                cell.className = cellData.className;
            }
            cell.innerHTML = cellData.content;
            
            if (cellData.title) {
                cell.title = cellData.title;
            }
            
            row.appendChild(cell);
        });
        
        // Add row to table body
        validationTableBody.appendChild(row);
    });
    
    // Show the validation table
    validationTableContainer.style.display = 'block';
    
    // Attach sort handlers
    attachValidationSortHandlers();

    // Attach filter listeners
    attachValidationFilterListeners();
}

// Function to attach sort handlers to validation table
function attachValidationSortHandlers() {
    document.querySelectorAll('#validation-table th[data-sort]').forEach(th => {
        // Remove existing handlers
        const newTh = th.cloneNode(true);
        th.parentNode.replaceChild(newTh, th);
        
        // Add new handler
        newTh.addEventListener('click', function() {
            const field = this.getAttribute('data-sort');
            
            if (field === sortFieldValidation) {
                sortDirectionValidation = sortDirectionValidation === 'asc' ? 'desc' : 'asc';
            } else {
                sortFieldValidation = field;
                sortDirectionValidation = 'asc';
            }
            
            // Update sort indicators
            document.querySelectorAll('#validation-table th').forEach(header => {
                header.classList.remove('sort-asc', 'sort-desc');
            });
            this.classList.add(sortDirectionValidation === 'asc' ? 'sort-asc' : 'sort-desc');
            
            // Re-display the table with the new sort
            fetchValidationData();
        });
    });
}

//Function to fetch validation data for sorting
function fetchValidationData() {
    const loadingSpinner = document.getElementById('loading-spinner');
    if (loadingSpinner) loadingSpinner.style.display = 'block';
    
    // Re-fetch from server or use cached data
    fetch(getApiUrl('/change-simulation/finalize-changes'), {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (loadingSpinner) loadingSpinner.style.display = 'none';
        
        if (data.success && data.result.final_commit_df) {
            displayValidationTable(data.result.final_commit_df);
        }
    })
    .catch(error => {
        if (loadingSpinner) loadingSpinner.style.display = 'none';
        console.error('Error fetching validation data:', error);
    });
}


function filterValidationTable() {
    const searchTerm = document.getElementById('validation-search').value.toLowerCase();
    const searchType = document.getElementById('validation-search-type').value;
    const tableRows = document.querySelectorAll('#validation-table-body tr');
    
    let visibleCount = 0;
    
    tableRows.forEach(row => {
        let match = false;
        
        if (searchType === 'contains' || searchType === 'not-contains') {
            // Search in all cells
            const rowText = row.textContent.toLowerCase();
            match = rowText.includes(searchTerm);
            
            // Invert match for not-contains
            if (searchType === 'not-contains') {
                match = !match;
            }
        } else if (searchType === 'contract-only') {
            // Search only in Contract Number column (6th column)
            const contractCell = row.cells[6]; 
            if (contractCell) {
                const contractText = contractCell.textContent.toLowerCase();
                match = contractText.includes(searchTerm);
            }
        } else if (searchType === 'validation-flag-only') {
            // Search only in Validation Flag column (3rd column)
            const validationFlagCell = row.cells[3]; 
            if (validationFlagCell) {
                const validationFlagText = validationFlagCell.textContent.toLowerCase();
                match = validationFlagText.includes(searchTerm);
            }
        } else if (searchType === 'item-only') {
            // Search only in Item column (4th column)
            const itemCell = row.cells[4];
            if (itemCell) {
                const itemText = itemCell.textContent.toLowerCase();
                match = itemText.includes(searchTerm);
            }
        }
        
        // Show/hide row based on match
        row.style.display = match ? '' : 'none';
        if (match) visibleCount++;
    });
}

// Function to clear validation search
function clearValidationSearch() {
    document.getElementById('validation-search').value = '';
    filterValidationTable();
}

// Attach event listeners for validation table
function attachValidationFilterListeners() {
    const searchInput = document.getElementById('validation-search');
    const searchType = document.getElementById('validation-search-type');
    const clearSearch = document.getElementById('validation-clear-search');
    
    if (searchInput && searchType && clearSearch) {
        searchInput.addEventListener('input', filterValidationTable);
        searchType.addEventListener('change', filterValidationTable);
        clearSearch.addEventListener('click', clearValidationSearch);
    }
}

// Function to count validation flags by type and update cards
function updateValidationFlagCounts(validationData) {
    let errorCount = 0;
    let warningCount = 0;
    let checkCount = 0;

    // Group validation data by flag type
    const errorItems = [];
    const warningItems = [];
    const checkItems = [];

    if (validationData && validationData.length > 0) {
        validationData.forEach(item => {
            const validationFlag = item['Validation Flag'] || '';
            if (validationFlag.startsWith('ERROR')) {
                errorCount++;
                errorItems.push(item);
            } else if (validationFlag.startsWith('WARNING')) {
                warningCount++;
                warningItems.push(item);
            } else if (validationFlag.startsWith('CHECK')) {
                checkCount++;
                checkItems.push(item);
            }
        });
    }

    // Update card counts
    document.getElementById('validation-error-count').textContent = errorCount;
    document.getElementById('validation-warning-count').textContent = warningCount;
    document.getElementById('validation-check-count').textContent = checkCount;

    // Update card border colors based on counts using CSS classes
    const errorCard = document.querySelector('.validation-error-card');
    const warningCard = document.querySelector('.validation-warning-card');
    const checkCard = document.querySelector('.validation-check-card');

    // Apply color coding:
    // - Count = 0: Green border (adds -zero class)
    // - Count > 0: Category-specific color (removes -zero class)
    //   - ERROR: Red border (default .validation-error-card)
    //   - WARNING: Orange border (default .validation-warning-card)
    //   - CHECK: Purple border (default .validation-check-card)
    
    if (errorCount === 0) {
        errorCard.classList.add('validation-error-card-zero');
    } else {
        errorCard.classList.remove('validation-error-card-zero');
    }
    
    if (warningCount === 0) {
        warningCard.classList.add('validation-warning-card-zero');
    } else {
        warningCard.classList.remove('validation-warning-card-zero');
    }
    
    if (checkCount === 0) {
        checkCard.classList.add('validation-check-card-zero');
    } else {
        checkCard.classList.remove('validation-check-card-zero');
    }

    // Store the filtered data for modal display
    window.validationFlagData = {
        error: errorItems,
        warning: warningItems,
        check: checkItems
    };

    // Show the cards container
    document.querySelector('.validation-flag-cards').style.display = 'grid';

    // Attach click handlers to the cards
    attachValidationCardClickHandlers();
}

// Function to attach click handlers to validation flag cards
function attachValidationCardClickHandlers() {
    // ERROR card
    document.getElementById('validation-error-count').closest('.change-stat-card').addEventListener('click', function() {
        showValidationFlagModal('ERROR', window.validationFlagData.error);
    });
    
    // WARNING card
    document.getElementById('validation-warning-count').closest('.change-stat-card').addEventListener('click', function() {
        showValidationFlagModal('WARNING', window.validationFlagData.warning);
    });
    
    // CHECK card
    document.getElementById('validation-check-count').closest('.change-stat-card').addEventListener('click', function() {
        showValidationFlagModal('CHECK', window.validationFlagData.check);
    });
}

// Function to show validation flag modal
function showValidationFlagModal(flagType, data) {
    // Set modal title
    document.getElementById('validationFlagModalLabel').textContent = `${flagType} Validation Items`;
    
    const tableBody = document.getElementById('validationFlagTableBody');
    const noDataMessage = document.getElementById('noValidationDataMessage');
    const tableContainer = document.getElementById('validationFlagTableContainer');
    
    // Clear existing content
    tableBody.innerHTML = '';
    
    if (!data || data.length === 0) {
        // Show no data message
        tableContainer.style.display = 'none';
        noDataMessage.style.display = 'block';
    } else {
        // Show table and populate data
        tableContainer.style.display = 'block';
        noDataMessage.style.display = 'none';
        
        data.forEach(item => {
            const row = document.createElement('tr');
            
            // Format contract price
            const contractPrice = item['Contract Price'] ? 
                '$' + parseFloat(item['Contract Price']).toFixed(2) : '';
            
            // Format validation flag for display
            const validationFlag = item['Validation Flag'] || '';
            const flagParts = validationFlag.split(' - ');
            const flagMessage = flagParts.length > 1 ? flagParts.slice(1).join(' - ') : '';
            
            // Format actions
            const formattedPrimaryAction = formatActionText(item['Primary Action'] || '');
            const formattedActualAction = formatActionText(item['Actual Action'] || '');
            
            row.innerHTML = `
                <td><span class="${getValidationFlagClass(validationFlag)}">${flagParts[0]}</span></td>
                <td>${flagMessage}</td>
                <td><span class="${getPrimaryActionClass(item['Primary Action'])}">${formattedPrimaryAction}</span></td>
                <td><span class="${getActualActionClass(item['Actual Action'])}">${formattedActualAction}</span></td>
                <td>${item['Item'] || ''}</td>
                <td>${item['Contract Number'] || ''}</td>
                <td>${item['ERP Vendor ID'] || ''}</td>
                <td>${item['Mfg Part Num'] || ''}</td>
                <td>${item['Vendor Part Num'] || ''}</td>
                <td>${item['UOM'] || ''}</td>
                <td>${item['QOE'] || ''}</td>
                <td>${contractPrice}</td>
                <td>${item['File Row'] || ''}</td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    // Show modal
    $('#validationFlagModal').modal('show');
}