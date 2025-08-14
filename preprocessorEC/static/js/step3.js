/*step3*/
/** * Step 3: Deduplication Policy and Resolution JavaScript
 * Handles deduplication policy selection, custom policy options,
 * drag-and-drop priority ranking,
 * deduplication resolution application,
 * and results display.
 * Also includes filtering, manual adjustments, and saving selections.
 * Supports both custom and manual resolution modes.
 */
document.addEventListener('DOMContentLoaded', function() {
    const resolutionStrategy = document.getElementById('resolution_strategy');
    const customPolicyOptions = document.getElementById('custom_policy_options');
    const priorityRankingArea = document.getElementById('priority_ranking_area');
    const draggableCards = document.querySelectorAll('.draggable-card');
    const preferenceSelects = document.querySelectorAll('.preference-select');
    const resolutionForm = document.getElementById('resolution-form');
    const deduplicationResults = document.getElementById('deduplication-results');
    const focusDifferencesBtn = document.getElementById('focus-differences-btn');
    const showAllBtn = document.getElementById('show-all-btn');
    const completeStepBtn = document.getElementById('complete-step-btn');
    const saveSelectionsBtn = document.getElementById('save-selections-btn');
    const intendedActionFilter = document.getElementById('intended-action-filter');
    const policySelect = document.getElementById('resolution_strategy');
    const policyDescriptions = document.querySelectorAll('.policy-description');

    document.getElementById('completion_resolution_strategy').value = document.getElementById('resolution_strategy').value;

    policySelect.addEventListener('change', function() {
        // Hide all descriptions
        policyDescriptions.forEach(desc => desc.classList.remove('active-description'));
        
        // Show the selected one
        const selectedDesc = document.getElementById(`${this.value}_desc`);
        if (selectedDesc) {
            selectedDesc.classList.add('active-description');
        }

        // add to the completion resolution strategy
        document.getElementById('completion_resolution_strategy').value = this.value;
    });

    if (saveSelectionsBtn) {
        saveSelectionsBtn.addEventListener('click', saveKeepSelections);
    }

    const completeStepForm = document.querySelector('form[action*="process_step"]');
    if (completeStepForm) {
        completeStepForm.addEventListener('submit', function(e) {
            e.preventDefault();

            showSpinner();
            
            saveKeepSelections();

            setTimeout(() => {
                this.submit();
            }, 1000); // Delay to allow for save completion
        });
    }

    if (focusDifferencesBtn) {
        focusDifferencesBtn.addEventListener('click', filterDifferenceGroups);
    }
    if (showAllBtn) {
        showAllBtn.addEventListener('click', showAllGroups);
    }
    if (intendedActionFilter) {
        intendedActionFilter.addEventListener('change', applyIntendedActionFilter);
    }


    let draggedItem = null;

    // --- Initial Setup ---
    toggleCustomOptions();
    updateHiddenInputs(); // Set initial values based on default order/prefs

    // --- Event Listeners ---
    resolutionStrategy.addEventListener('change', toggleCustomOptions);

    draggableCards.forEach(card => {
        card.addEventListener('dragstart', handleDragStart);
        card.addEventListener('dragend', handleDragEnd);
    });

    priorityRankingArea.addEventListener('dragover', handleDragOver);
    priorityRankingArea.addEventListener('dragenter', handleDragEnter);
    priorityRankingArea.addEventListener('dragleave', handleDragLeave);
    priorityRankingArea.addEventListener('drop', handleDrop);

    preferenceSelects.forEach(select => {
        select.addEventListener('change', updateHiddenInputs);
    });
    
    // Form submission handler
    resolutionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        applyResolution();
    });

    // --- Functions ---
    function toggleCustomOptions() {
        // Show custom policy options for both 'custom' and 'manual' modes
        if (resolutionStrategy.value === 'custom' || resolutionStrategy.value === 'manual') {
            customPolicyOptions.style.display = 'block';
            
            // Update the header text based on selected mode
            const policyHeader = customPolicyOptions.querySelector('h4');
            if (policyHeader) {
                if (resolutionStrategy.value === 'custom') {
                    policyHeader.textContent = 'Custom Deduplication Policy';
                } else {
                    policyHeader.textContent = 'Manual Deduplication: Initial Priority Settings';
                }
            }
            
            // Update the description text based on selected mode
            const policyDesc = customPolicyOptions.querySelector('p.text-muted');
            if (policyDesc) {
                if (resolutionStrategy.value === 'custom') {
                    policyDesc.textContent = 'Drag and drop the parameters below to set their priority order (Top = Highest Priority). Select/Change your preference within each card.';
                } else {
                    policyDesc.textContent = 'Set initial priorities as a first pass. After applying, you can manually adjust individual selections using checkboxes in the results table.';
                }
            }
            
            // Show explanation tooltip on first display for each mode
            const storageKey = resolutionStrategy.value === 'custom' ? 'customPolicyExplained' : 'manualPolicyExplained';
            if (!sessionStorage.getItem(storageKey)) {
                if (resolutionStrategy.value === 'custom') {
                    alert('Custom Policy Guide: Drag and drop parameters to rank them (top is highest priority). Select preferences within each card.');
                } else {
                    alert('Manual Mode Guide: First set general priorities with drag and drop, then after applying you can override individual selections with checkboxes.');
                }
                sessionStorage.setItem(storageKey, 'true');
            }
            
            updateHiddenInputs(); // Ensure inputs are updated when shown
        } else {
            customPolicyOptions.style.display = 'none';
        }
        
        // Show/hide Save Selections button based on policy
        const saveSelectionsContainer = document.getElementById('save-selections-container');
        if (saveSelectionsContainer) {
            saveSelectionsContainer.style.display = resolutionStrategy.value === 'manual' ? 'block' : 'none';
        }

        // Update policy description visibility
        const policyDescriptions = document.querySelectorAll('.policy-description');
        policyDescriptions.forEach(desc => desc.classList.remove('active-description'));
        const selectedDesc = document.getElementById(`${resolutionStrategy.value}_desc`);
        if (selectedDesc) {
            selectedDesc.classList.add('active-description');
        }
    }

    // Draggable functions remain unchanged...
    // handleDragStart, handleDragEnd, handleDragOver, handleDragEnter, handleDragLeave, handleDrop, getDragAfterElement, updateHiddenInputs

    function handleDragStart(e) {
        draggedItem = this;
        this.classList.add('dragging'); // Add class immediately
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', this.id); // Pass the id
    }

    function handleDragEnd() {
        if (draggedItem) {
            draggedItem.classList.remove('dragging');
            draggedItem = null;
        }
        priorityRankingArea.classList.remove('drag-over');
    }

    function handleDragOver(e) {
        e.preventDefault(); // Necessary to allow dropping
        e.dataTransfer.dropEffect = 'move';
        
        // Don't manipulate DOM during dragOver - just visual feedback
        priorityRankingArea.classList.add('drag-over');
    }

    function handleDragEnter(e) {
        e.preventDefault(); // Prevent default behavior
        // Optional: Add more specific visual feedback if needed
    }

    function handleDragLeave() {
            priorityRankingArea.classList.remove('drag-over'); // Remove highlight when leaving
    }

    function handleDrop(e) {
        e.preventDefault();
        priorityRankingArea.classList.remove('drag-over');

        const id = e.dataTransfer.getData('text/plain');
        const draggable = document.getElementById(id);
        
        if (!draggable) return;
        
        const afterElement = getDragAfterElement(priorityRankingArea, e.clientY);
        
        // Perform the actual DOM manipulation on drop, not during dragover
        if (afterElement == null) {
            priorityRankingArea.appendChild(draggable);
        } else {
            priorityRankingArea.insertBefore(draggable, afterElement);
        }
        
        draggable.classList.remove('dragging');
        draggedItem = null;
        
        // Update priorities after reordering
        updateHiddenInputs();
    }

    function getDragAfterElement(container, y) {
        // Only consider cards that are not currently being dragged
        const draggableElements = [...container.querySelectorAll('.draggable-card:not(.dragging)')];
        
        return draggableElements.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            const offset = y - box.top - box.height / 2;
            
            // If mouse position is above middle of element and this offset is smaller than previous
            if (offset < 0 && offset > closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY }).element;
    }

    function updateHiddenInputs() {
        const cardsInOrder = priorityRankingArea.querySelectorAll('.draggable-card');
        cardsInOrder.forEach((card, index) => {
            const priority = index + 1; // Priority 1-4 based on order
            const parameterId = card.id.replace('card_', ''); // e.g., 'contract_source'
            const preferenceSelect = card.querySelector('.preference-select');
            const preferenceValue = preferenceSelect && preferenceSelect.value ? preferenceSelect.value : ''; // Get selected preference

            // Update hidden inputs
            document.getElementById(`${parameterId}_priority`).value = priority;
            document.getElementById(`${parameterId}_preference`).value = preferenceValue;
            
            // Update rank indicator
            const rankIndicator = card.querySelector('.rank-indicator');
            if (rankIndicator) {
                rankIndicator.textContent = priority;
            }
        });
    }
    
    // New functions for deduplication
    function applyResolution() {
        // Show loading indicator
        showSpinner();
        
        // Get form data
        const formData = new FormData(resolutionForm);
        
        // Send AJAX request
        fetch(getApiUrl('/duplicate-detection/apply-resolution'), {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            
            if (data.success) {
                // Show success message
                showAlert('success', data.message);
                
                // Show results section
                deduplicationResults.style.display = 'block';
                
                // Update statistics
                document.getElementById('stats-total-items').textContent = data.summary.total_items;
                document.getElementById('stats-unique-items').textContent = data.summary.unique_duplicates;
                document.getElementById('stats-kept-ccx').textContent = data.summary.kept_ccx;
                document.getElementById('stats-kept-uploaded').textContent = data.summary.kept_uploaded;
                
                // Show/hide Save Selections button based on policy
                const saveSelectionsContainer = document.getElementById('save-selections-container');
                if (saveSelectionsContainer) {
                    saveSelectionsContainer.style.display = resolutionStrategy.value === 'manual' ? 'block' : 'none';
                }
                // Fetch and display results
                fetchDeduplicationResults();

                // Set "Show All" as default active button
                document.getElementById('focus-differences-btn').classList.remove('active');
                document.getElementById('show-all-btn').classList.add('active');
                
                // Scroll to results
                deduplicationResults.scrollIntoView({ behavior: 'smooth' });
            } else {
                showAlert('danger', data.message);
            }
        })
        .catch(error => {
            hideSpinner();
            showAlert('danger', 'An error occurred: ' + error.message);
        });
    }
    
    function fetchDeduplicationResults() {
        fetch(getApiUrl('/duplicate-detection/get-deduplication-results'))
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // for manual mode, initialize kept stats to 0
                if (resolutionStrategy.value === 'manual') {
                    if(data.data.summary) {
                        data.data.summary.kept_ccx = 0;
                        data.data.summary.kept_uploaded = 0;

                        document.getElementById('stats-kept-ccx').textContent = '0';
                        document.getElementById('stats-kept-uploaded').textContent = '0';
                    }
                }
                populateResultsTable(data.data);
                if (resolutionStrategy.value === 'manual') {
                    updateManualModeStats();
                }
            }
        })
        .catch(error => {
            console.error('Error fetching deduplication results:', error);
        });
    }

    function updateManualModeStats() {
        if (resolutionStrategy.value !== 'manual') return;
        
        let keptCCX = 0;
        let keptTP = 0;
        
        // Count checked items by dataset
        document.querySelectorAll('.keep-checkbox:checked').forEach(checkbox => {
            // Get the dataset from the table cell (4th column contains Dataset)
            const row = checkbox.closest('tr');
            const dataset = row.cells[3].textContent.trim();
            
            if (dataset === 'CCX') {
                keptCCX++;
            } else if (dataset === 'TP') {
                keptTP++;
            }
        });
        
        // Update the stats display
        document.getElementById('stats-kept-ccx').textContent = keptCCX;
        document.getElementById('stats-kept-uploaded').textContent = keptTP;
    }
    
    function populateResultsTable(data) {
        const tableBody = document.getElementById('results-table-body');
        tableBody.innerHTML = '';
        
        if (!data.stacked_data || data.stacked_data.length === 0) {
            // Update colspan for empty table message
            tableBody.innerHTML = '<tr><td colspan="13" class="text-center">No results found</td></tr>';
            return;
        }
        
        // Get upsert and expire sets from the data
        const upsertSet = new Set(data.upsert_set || []);
        const expireSet = new Set(data.expire_set || []);

        // Store globally for filter functions
        window.currentUpsertSet = upsertSet;
        window.currentExpireSet = expireSet;
        
        // Group data by File Row
        const groupedData = {};
        data.stacked_data.forEach(item => {
            const groupId = item['File Row'];
            if (!groupedData[groupId]) {
                groupedData[groupId] = [];
            }
            groupedData[groupId].push(item);
        });
        
        // Get group IDs and sort
        const groupIds = Object.keys(groupedData);
        
        // Show all groups, not just top 10
        const displayGroups = groupIds;
        
        // Function to format price as currency
        function formatCurrency(value) {
            if (value === null || value === undefined || value === '') {
                return 'N/A';
            }
            // Parse as float and format with $ and 2 decimal places
            return '$' + parseFloat(value).toFixed(2);
        }
        
        // get CURRENT POLICY
        const currentPolicy = document.getElementById('resolution_strategy').value;

        // Add rows for each group
        displayGroups.forEach((groupId, groupIndex) => {
            // Sort by group ID (File Row)// Sort by rank within group
            groupedData[groupId].sort((a, b) => a['Rank'] - b['Rank']);

            // Store reference value
            const rank1Item = groupedData[groupId].find(i => i['Rank'] === 1);
            const rank1QOE = rank1Item ? rank1Item['QOE'] : null;
            const rank1UOM = rank1Item ? rank1Item['UOM'] : null;
            const rank1EAPrice = rank1Item ? rank1Item['EA Price'] : null;
            
            // Determine row styling based on intended action
            const fileRowNum = parseInt(groupId);
            const isUpsert = upsertSet.has(fileRowNum);
            const isExpire = expireSet.has(fileRowNum);

            // Add group items
            groupedData[groupId].forEach((item, itemIndex) => {
                const row = document.createElement('tr');
                
                // Add highlighting class based on intended action and rank
                if (item['Rank'] === 1) {
                    if (isExpire) {
                        row.style.backgroundColor = '#f8d7da'; // Light red for expire items
                    } else if (isUpsert) {
                        row.classList.add('rank-1'); // Keep existing green styling for upsert items
                    }
                }

                // Add Group cell for EVERY row
                const groupCell = document.createElement('td');
                groupCell.textContent = groupId;
                groupCell.classList.add('group-cell');

                // Only make the text visible for the first row in each group
                if (itemIndex > 0) {
                    groupCell.style.color = 'transparent'; // Hide text but keep cell structure
                }
                row.appendChild(groupCell);
                
                // Add Rank cell
                const rankCell = document.createElement('td');
                rankCell.textContent = item['Rank'] || '';
                row.appendChild(rankCell);
                
                // --- Add Keep Checkbox Cell (now third column) ---
                const keepCell = document.createElement('td');
                keepCell.classList.add('keep-cell');
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.classList.add('keep-checkbox');
                // Add data attributes to identify the item for potential manual handling later
                checkbox.dataset.fileRow = item['File Row'];
                checkbox.dataset.pairId = item['Pair ID']; // Make sure 'Pair ID' exists in your data

                
                // Check Rank 1 if policy is not manual
                if (item['Rank'] === 1) {
                    checkbox.checked = true;
                }

                // Disable checkbox if policy is not manual
                if (currentPolicy !== 'manual') {
                    checkbox.disabled = true;
                } else {
                    // Add event listener for manual mode
                    checkbox.addEventListener('change', function() {
                        const row = this.closest('tr');
                        const groupID = this.dataset.fileRow;
                        const dataset = item['Dataset'] || ''; // Get dataset name for the row

                        if (this.checked) {
                            // uncheck all other checkboxes in the same group
                            document.querySelectorAll('.keep-checkbox').forEach(cb => {
                                if (cb !== this && cb.dataset.fileRow === groupID) {
                                    cb.checked = false;
                                    const otherRow = cb.closest('tr');
                                    if (otherRow) {
                                        otherRow.classList.remove('rank-1'); // Remove class from other rows in the group
                                    }
                                }
                            });
                            
                            row.classList.add('rank-1'); // Add class to highlight selected row
                            if (currentPolicy === 'manual') {
                                // count selected checkboxes by dataset
                                updateManualModeStats();
                            }
                        } else {
                            row.classList.remove('rank-1'); // Remove class if unchecked
                            if (currentPolicy === 'manual') {
                                // count selected checkboxes by dataset
                                updateManualModeStats();
                            }
                        }
                    });
                }

                keepCell.appendChild(checkbox);
                row.appendChild(keepCell);
                // --- End Keep Checkbox Cell ---
                
                // Process description to ensure consistent truncation
                const description = item['Description'] || '';
                const truncatedDescription = description.length > 40 ? 
                    description.substring(0, 40) + '...' : description;
                
                // Format prices with $ and 2 decimal places
                const formattedContractPrice = formatCurrency(item['Contract Price']);
                const formattedEAPrice = formatCurrency(item['EA Price']);
                
                // Add remaining cells in the new order
                const cells = [
                    ['Dataset', item['Dataset']],
                    ['Contract Number', item['Contract Number']],
                    ['Mfg Part Num', item['Mfg Part Num']],
                    ['UOM', item['UOM']],
                    ['QOE', item['QOE']],
                    ['Contract Price', formattedContractPrice],
                    ['EA Price', formattedEAPrice],
                    ['Effective Date', item['Effective Date']],
                    ['Expiration Date', item['Expiration Date']],
                    ['Description', truncatedDescription]
                ];
                
                cells.forEach(([field, value]) => {
                    const cell = document.createElement('td');
                    cell.textContent = value || '';
                    
                    // Add title attribute to Description column for hover tooltip
                    if (field === 'Description') {
                        cell.setAttribute('title', description);
                        cell.classList.add('description-cell');
                    }

                    // highlight the cell if it is different from rank-1 int he same group
                    if (item['Rank'] !== 1 && rank1Item) {
                        if (field === 'QOE' && item['QOE'] !== rank1QOE && rank1QOE !== null) {
                            cell.classList.add('different-value');
                        } else if (field === 'UOM' && item['UOM'] !== rank1UOM && rank1UOM !== null) {
                            cell.classList.add('different-value');
                        } else if (field === 'EA Price' && rank1EAPrice !== null && item['EA Price'] !== null) {
                            // For EA Price, apply different highlighting based on price ratio
                            const itemPrice = parseFloat(item['EA Price']);
                            const rank1Price = parseFloat(rank1EAPrice);
                            if (Math.abs(itemPrice - rank1Price) > 0.01) {
                                const priceRatio = parseFloat(item['EA Price']) / parseFloat(rank1EAPrice);
                                
                                // Check if price is significantly different (>2x or <0.5x)
                                if (priceRatio >= 2 || priceRatio < 0.5) {
                                    cell.classList.add('significant-price-difference');
                                } else {
                                    cell.classList.add('moderate-price-difference');
                                }
                            }
                        }
                    }
                    
                    row.appendChild(cell);
                });
                
                tableBody.appendChild(row);
            });
            
            // Add separator between groups (except after the last group)
            if (groupIndex < displayGroups.length - 1) {
                const divider = document.createElement('tr');
                divider.classList.add('group-divider');
                // Update colspan for divider (still 13 columns total)
                divider.innerHTML = '<td colspan="13"></td>';
                tableBody.appendChild(divider);
            }
        });
    }

    function saveKeepSelections() {
        showSpinner();
        
        const currentPolicy = document.getElementById('resolution_strategy').value;
        console.log('Sending policy:', currentPolicy);
        
        // Collect all checkbox states
        const keepSelections = [];
        document.querySelectorAll('.keep-checkbox').forEach(checkbox => {
            const fileRow = checkbox.dataset.fileRow;
            const pairId = checkbox.dataset.pairId;
            
            if (fileRow && pairId) {
                // Find the row to get the dataset value
                const row = checkbox.closest('tr');
                const datasetCell = row.querySelector('td:nth-child(4)'); // 4th column has Dataset
                const dataset = datasetCell ? datasetCell.textContent.trim() : '';
                
                // Determine keep status based on mode
                let keepStatus = checkbox.checked;
                
                if (currentPolicy !== 'manual') {
                    // For non-manual modes, use rank to determine keep status
                    const rankCell = row.querySelector('td:nth-child(2)'); // 2nd column has Rank
                    keepStatus = (rankCell && rankCell.textContent.trim() === '1');
                }
                
                // Only push ONCE to the array
                keepSelections.push({
                    file_row: parseInt(fileRow),
                    pair_id: pairId,
                    dataset: dataset,
                    keep: keepStatus // Fixed property name
                });
            }
        });
        
        console.log('Keep selections to send:', keepSelections); // Debug log
        console.log('Current policy:', currentPolicy); // Debug policy
        
        // Send to server
        fetch(getApiUrl('/duplicate-detection/update-keep-status'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                keep_selections: keepSelections,
                policy: currentPolicy // Include the current policy in the request
            })
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            
            if (data.success) {                    
                // Update summary stats with the new values
                document.getElementById('stats-kept-ccx').textContent = data.summary.kept_ccx;
                document.getElementById('stats-kept-uploaded').textContent = data.summary.kept_uploaded;
                // show modal
                alert('Selections saved successfully!');

            } else {
                showAlert('danger', data.message);
            }
        })
        .catch(error => {
            hideSpinner();
            showAlert('danger', 'Error saving selections: ' + error.message);
            console.error('Error saving keep status:', error);
        });
    }

    // Function to apply intended action filter
    function applyIntendedActionFilter() {
        const selectedAction = intendedActionFilter.value;
        const tableBody = document.getElementById('results-table-body');
        const rows = tableBody.querySelectorAll('tr');
        
        // Get upsert and expire sets from the data (store these globally when data is loaded)
        const upsertSet = window.currentUpsertSet || new Set();
        const expireSet = window.currentExpireSet || new Set();
        
        // Track which groups to show
        const visibleGroups = new Set();
        
        // First pass: determine which groups should be visible based on intended action
        rows.forEach(row => {
            if (row.classList.contains('group-divider')) return;
            
            const groupCell = row.querySelector('td:first-child');
            const groupId = groupCell ? groupCell.textContent.trim() : null;
            
            if (groupId) {
                const fileRowNum = parseInt(groupId);
                const isUpsert = upsertSet.has(fileRowNum);
                const isExpire = expireSet.has(fileRowNum);
                
                if (selectedAction === 'all' || 
                    (selectedAction === 'upsert' && isUpsert) ||
                    (selectedAction === 'expire' && isExpire)) {
                    visibleGroups.add(groupId);
                }
            }
        });
        
        // Second pass: show/hide rows based on group visibility
        let currentGroup = null;
        let hideCurrentGroup = false;
        
        rows.forEach(row => {
            if (row.classList.contains('group-divider')) {
                row.style.display = hideCurrentGroup ? 'none' : '';
                return;
            }
            
            const groupCell = row.querySelector('td:first-child');
            const groupId = groupCell ? groupCell.textContent.trim() : null;
            
            if (groupId && groupId !== currentGroup) {
                currentGroup = groupId;
                hideCurrentGroup = !visibleGroups.has(groupId);
            }
            
            row.style.display = hideCurrentGroup ? 'none' : '';
        });
        
        // Reset filter buttons to default state when intended action filter changes
        document.getElementById('focus-differences-btn').classList.remove('active');
        document.getElementById('show-all-btn').classList.add('active');
    }

    // Function to filter and only show groups containing differences
    function filterDifferenceGroups() {
        const selectedAction = intendedActionFilter.value;
        const tableBody = document.getElementById('results-table-body');
        const rows = tableBody.querySelectorAll('tr');
        
        // Get upsert and expire sets
        const upsertSet = window.currentUpsertSet || new Set();
        const expireSet = window.currentExpireSet || new Set();
        
        // Track which groups have differences AND match intended action filter
        const groupsWithDifferences = new Set();
        
        // First pass: identify groups with differences that match intended action filter
        rows.forEach(row => {
            if (row.classList.contains('group-divider')) return;
            
            const groupCell = row.querySelector('td:first-child');
            const groupId = groupCell ? groupCell.textContent.trim() : null;
            
            if (groupId) {
                const fileRowNum = parseInt(groupId);
                const isUpsert = upsertSet.has(fileRowNum);
                const isExpire = expireSet.has(fileRowNum);
                
                // Check if group matches intended action filter
                const matchesIntendedAction = selectedAction === 'all' || 
                                            (selectedAction === 'upsert' && isUpsert) ||
                                            (selectedAction === 'expire' && isExpire);
                
                if (matchesIntendedAction) {
                    const hasDifference = row.querySelector('.different-value, .moderate-price-difference, .significant-price-difference');
                    if (hasDifference) {
                        groupsWithDifferences.add(groupId);
                    }
                }
            }
        });
        
        // Second pass: hide groups without differences or that don't match intended action
        let currentGroup = null;
        let hideCurrentGroup = false;
        
        rows.forEach(row => {
            if (row.classList.contains('group-divider')) {
                row.style.display = hideCurrentGroup ? 'none' : '';
                return;
            }
            
            const groupCell = row.querySelector('td:first-child');
            const groupId = groupCell ? groupCell.textContent.trim() : null;
            
            if (groupId && groupId !== currentGroup) {
                currentGroup = groupId;
                hideCurrentGroup = !groupsWithDifferences.has(groupId);
            }
            
            row.style.display = hideCurrentGroup ? 'none' : '';
        });
        
        // Update button state
        document.getElementById('focus-differences-btn').classList.add('active');
        document.getElementById('show-all-btn').classList.remove('active');
    }

    // Function to show all groups
    function showAllGroups() {
        // Apply intended action filter instead of showing absolutely all
        applyIntendedActionFilter();
        
        // Update button state
        document.getElementById('focus-differences-btn').classList.remove('active');
        document.getElementById('show-all-btn').classList.add('active');
    }
    
    // Utility functions
    function showSpinner() {
        // Use the existing spinner element instead of creating a new one
        const spinner = document.getElementById('loading-spinner');
        if (spinner) {
            // Show the spinner
            spinner.style.display = 'flex';
        }
    }

    function hideSpinner() {
        const spinner = document.getElementById('loading-spinner');
        if (spinner) {
            spinner.style.display = 'none';
        }
    }
    
    function showAlert(type, message, timeout = 5000) {
        // Remove existing alerts
        // Use the global alert function from layout.html
        window.showGlobalAlert(type, message, timeout);
    }
});

