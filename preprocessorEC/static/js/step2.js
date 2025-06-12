document.addEventListener('DOMContentLoaded', function() {
    
    // Action buttons
    const selectAllBtn = document.getElementById('select-all-btn');
    const deselectAllBtn = document.getElementById('deselect-all-btn');
    const saveSelectionsBtn = document.getElementById('save-selections-btn');

    const batchIncludeBtn = document.getElementById('batch-include-btn');
    const batchExcludeBtn = document.getElementById('batch-exclude-btn');
    const batchContractActions = document.getElementById('batch-contract-actions');


    // Contract-Level Duplication Review Elements
    const loadingOverlaySpinner = document.getElementById('loading-overlay-spinner');
    const startAnalysisBtn = document.getElementById('start-analysis-btn');
    const contractCardsContainer = document.getElementById('contract-cards-container');
    const contractClearSearchBtn = document.getElementById('contract-clear-search-btn');
    const continueBtn = document.getElementById('continue-btn');
    const contractSearch = document.getElementById('contract-search');
    const searchTypeSelect = document.getElementById('search-type');
    const modal = document.getElementById('item-modal');
    const closeBtn = modal.querySelector('.close-btn');

    // this is the top level item comparison container
    const itemComparisonContainer = document.getElementById('item-comparison-container');
    // this is the container for the confidence cards
    const comparisonResultsContainer = document.getElementById('comparison-results-container');
    // Item-Level Comparison Code
    const clearSearchBtn = document.getElementById('clear-search-btn');
    const itemSearchInput = document.getElementById('item-search');
    const itemSearchType = document.getElementById('item-search-type');
    const itemsDetailView = document.getElementById('items-detail-view');
    const detailTitle = document.getElementById('detail-title');
    const itemsTable = document.getElementById('items-table');
    const itemsTbody = document.getElementById('items-tbody');
    const noMatchesMessage = document.getElementById('no-matches-message');

    // Progress bar elements
    const loadingOverlayProgress = document.getElementById('loading-overlay-progress');
    const progressBar = document.getElementById('progress-bar');
    const progressPercentage = document.getElementById('progress-percentage');
    const progressMessage = document.getElementById('progress-message');
    const itemsProcessed = document.getElementById('items-processed');


    // Add these event listeners for batch buttons
    batchIncludeBtn.addEventListener('click', function() {
        batchToggleContracts(true);
    });

    batchExcludeBtn.addEventListener('click', function() {
        batchToggleContracts(false);
    });
    
    // Function to handle batch include/exclude
    let state = {
        allContracts: [],
        includedContracts: [],
        excludedContracts: [],
        isLoaded: false
    };
    
    // 2.2 Item-Level Comparison Variables
    let currentItems = [];
    let currentLevel = null;
    let sortField = null;
    let sortDirection = 'asc';

    // start analysis
    startAnalysisBtn.addEventListener('click', function() {
    // Show loading overlay
    loadingOverlaySpinner.style.display = 'flex';
    
    // Reset local state arrays
    state.includedContracts = [];
    state.excludedContracts = [];
    state.allContracts = [];
    
    // Start the duplication analysis
    fetch(getApiUrl('/duplicate-detection/process-duplicates'), {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Store contracts
                state.allContracts = data.contracts;
                
                // Initialize all contracts as included by default
                state.includedContracts = data.contracts.map(contract => contract.contract_number);
                
                // Send the included contracts to the server
                return fetch(getApiUrl('/duplicate-detection/initialize-included-contracts'), {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        contract_numbers: state.includedContracts
                    })
                });
            } else {
                throw new Error(data.message);
            }
        })
        .then(response => response.json())
        .then(data => {
            // Handle response from initialize-included-contracts
            if (data.success) {
                // Display contracts with all initially included
                displayContracts(state.allContracts);
                
                // Show review actions
                document.querySelector('.review-actions').style.display = 'block';
                document.querySelector('.contracts-status-section').style.display = 'block';
                batchContractActions.style.display = 'flex';
            } else {
                throw new Error(data.message);
            }
            
            // Hide loading overlay
            loadingOverlaySpinner.style.display = 'none';
        })
        .catch(error => {
            loadingOverlaySpinner.style.display = 'none';
            alert('Error: ' + error);
        });
    });

    loadingOverlaySpinner.style.display = 'flex';

    // 1. First, fetch the contract state
    fetch(getApiUrl('/duplicate-detection/get-contract-state'))
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                state.includedContracts = data.included_contracts || [];
                state.excludedContracts = data.excluded_contracts || [];
                
                console.log("Loaded contract state:", {
                    included: state.includedContracts.length,
                    excluded: state.excludedContracts.length
                });
                
                // 2. If we have included contracts, load the contract data but DON'T display yet
                if (state.includedContracts.length > 0) {
                    return loadContractData();
                } else {
                    state.isLoaded = true;
                    return { contracts: [] };
                }
            } else {
                throw new Error("Failed to load contract state: " + data.message);
            }
        })
        .then(data => {
            // 3. Store the contracts and mark as loaded
            if (data && data.contracts) {
                state.allContracts = data.contracts;
            }
            state.isLoaded = true;
            
            // Don't display contracts automatically, just show the empty placeholder
            // displayContracts();
            
            // Instead, ALWAYS HIDE the UI elements on page load regardless of state
            document.querySelector('.review-actions').style.display = 'none';
            document.querySelector('.contracts-status-section').style.display = 'none';
            batchContractActions.style.display = 'none';
            updateIncludedContractsTags();
        })
        .catch(error => {
            console.error("Error initializing page:", error);
            state.isLoaded = true;
        })
        .finally(() => {
            // Hide the loading spinner regardless of outcome
            loadingOverlaySpinner.style.display = 'none';
        });

    // Function to load contract data - returns a Promise
    function loadContractData() {
        return fetch(getApiUrl('/duplicate-detection/process-duplicates'), {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                throw new Error(data.message);
            }
            return data;
        });
    }
    

    function initializeIncludedContracts() {
        console.log("Saving included contracts:", state.includedContracts);
        
        // Send included contracts to the server
        fetch(getApiUrl('/duplicate-detection/initialize-included-contracts'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contract_numbers: state.includedContracts
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log("Successfully saved included contracts:", data);
                state.includedContracts = data.included_contracts;
                state.excludedContracts = data.excluded_contracts;
            } else {
                console.error("Error saving included contracts");
            }
        })
        .catch(error => {
            console.error("Error:", error);
        });
    }


    contractClearSearchBtn.addEventListener('click', function() {
        // Clear the search input
        contractSearch.value = '';
        // Reset search type to "contains"
        searchTypeSelect.value = 'contains';
        // Trigger filtering to show all contracts
        filterContracts();
    });

    
    function displayContracts(contractsToDisplay) {
        contractCardsContainer.innerHTML = '';
        
        // Get contracts to display - use parameter if provided, otherwise filter all contracts
        const contracts = contractsToDisplay || filterContracts();
            
            if (contracts.length === 0) {
                // Choose the appropriate message based on our state
                if (state.allContracts.length === 0) {
                    // No contracts exist at all
                    contractCardsContainer.innerHTML = `
                        <div class="no-results">
                            <h4>No matching contracts found on CCX</h4>
                            <p>Mostly there is no duplication for your uploaded items on CCX, we can skip step2 and 3.</p>
                            <div class="action-buttons" style="margin-top: 20px; text-align: center;">
                                <form method="POST" action="{{ url_for('common.process_step', step_id=4) }}">
                                    <input type="hidden" name="skip_steps" value="2,3">
                                    <input type="hidden" name="step_completed" value="true">
                                    <button type="submit" class="btn btn-success">Skip to Step 4</button>
                                </form>
                            </div>
                        </div>
                    `;
                } else if (state.includedContracts.length === 0 && state.isLoaded) {
                    // We have contracts but none are included
                    contractCardsContainer.innerHTML = `
                        <div class="no-results">
                            <h4>No contracts are currently included</h4>
                            <p>All contracts have been excluded. You can include some contracts or skip to step 4.</p>
                            <div class="action-buttons" style="margin-top: 20px; text-align: center;">
                                <form method="POST" action="{{ url_for('common.process_step', step_id=4) }}">
                                    <input type="hidden" name="skip_steps" value="2,3">
                                    <input type="hidden" name="step_completed" value="true">
                                    <button type="submit" class="btn btn-success">Skip to Step 4</button>
                                </form>
                            </div>
                        </div>
                    `;
                } else {
                    // We're just filtering and no contracts match the filter
                    contractCardsContainer.innerHTML = `
                        <div class="no-results">
                            <h4>No contracts match your search criteria</h4>
                            <p>Try changing your filter or clear the search to see all available contracts.</p>
                        </div>
                    `;
                }
                
                // Set visibility of UI sections
                document.querySelector('.review-actions').style.display = 'none';
                document.querySelector('.contracts-status-section').style.display = 'block';
                batchContractActions.style.display = 'flex';
                return;
            }

            // Automatically include all contracts by default if not explicitly excluded
            contracts.forEach(contract => {
                if (!state.excludedContracts.includes(contract.contract_number) && 
                    !state.includedContracts.includes(contract.contract_number)) {
                    state.includedContracts.push(contract.contract_number);
                }
            });
            
            // Sync the included contracts with the server
            fetch(getApiUrl('/duplicate-detection/initialize-included-contracts'), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    contract_numbers: state.includedContracts
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update local variables with server data
                    state.includedContracts = data.included_contracts;
                    state.excludedContracts = data.excluded_contracts;
                } else {
                    console.error('Error initializing contracts:', data.message);
                }
            })
            .catch(error => {
                console.error('Error initializing contracts:', error);
            });
            
            // Update included contracts tags display
            updateIncludedContractsTags();
            
            contracts.forEach(contract => {
                const isExcluded = state.excludedContracts.includes(contract.contract_number);
                
                const card = document.createElement('div');
                card.className = `contract-card ${isExcluded ? 'excluded' : 'selected'}`;
                card.dataset.contractNumber = contract.contract_number;
                
                card.innerHTML = `
                    <div class="card-header">
                        <div>
                            <h4 class="card-title">${contract.contract_number}</h4>
                            <div class="card-subtitle">${contract.contract_description || 'No description'}</div>
                            <div class="card-subtitle">Manufacturer: <strong>${contract.manufacturer_name || 'Unknown'}</strong></div>
                        </div>
                        <div class="status-indicator">
                            <span class="status-text ${isExcluded ? 'status-excluded' : 'status-included'}">
                                ${isExcluded ? 'Excluded' : 'Included'}
                            </span>
                            <button type="button" class="toggle-action-btn" data-contract="${contract.contract_number}">
                                ${isExcluded ? 'Include' : 'Exclude'}
                            </button>
                        </div>
                    </div>
                    <div class="card-stats">
                        <div class="card-stat">
                            <div class="stat-value">${contract.total_matches}</div>
                            <div class="stat-label">Total Matches</div>
                        </div>
                        <div class="card-stat">
                            <div class="stat-value">${contract.exact_matches}</div>
                            <div class="stat-label">Exact MFN Matches</div>
                        </div>
                        <div class="card-stat">
                            <div class="stat-value">${contract.total_line_count_ccx || 'N/A'}</div>
                            <div class="stat-label">Total CCX Lines</div>
                        </div>
                    </div>
                `;
                
                contractCardsContainer.appendChild(card);
            });
            
            // Add click event to card for details view
            document.querySelectorAll('.contract-card').forEach(card => {
                card.addEventListener('click', function(e) {
                    // Ignore clicks on the toggle button
                    if (e.target.classList.contains('toggle-action-btn') || 
                        e.target.closest('.toggle-action-btn')) {
                        return;
                    }
                    
                    // Show contract details
                    showContractDetails(this.dataset.contractNumber);
                });
            });
            
            // Add click event to toggle buttons - fixed to use proper selector
            document.querySelectorAll('.toggle-action-btn').forEach(btn => {
                btn.addEventListener('click', function(e) {
                    e.stopPropagation(); // Prevent opening the modal
                    
                    const contractNum = this.dataset.contract;
                    const isExcluded = state.excludedContracts.includes(contractNum);
                    
                    // Toggle inclusion state
                    toggleContractInclusion(contractNum, isExcluded);
                });
            });

            document.querySelector('.review-actions').style.display = 'block';
            document.querySelector('.contracts-status-section').style.display = 'block';
            batchContractActions.style.display = 'flex';
        }

    // Add this new function to update the tags display
    function updateIncludedContractsTags() {
        const tagsContainer = document.getElementById('included-contracts-tags');
        if (!tagsContainer) return;
        
        tagsContainer.innerHTML = '';
        
        if (state.includedContracts.length === 0) {
            tagsContainer.innerHTML = '<span class="empty-tag">No contracts included</span>';
            return;
        }
        
        state.includedContracts.forEach(contractNum => {
            const tag = document.createElement('span');
            tag.className = 'contract-tag';
            tag.textContent = contractNum;
            tagsContainer.appendChild(tag);
        });
    }

    // Update the toggle function to refresh the tags
    function toggleContractInclusion(contractNum, currentlyExcluded) {
        // Show loading overlay
        loadingOverlaySpinner.style.display = 'flex';
        
        // Send request to update inclusion status
        fetch(getApiUrl('/duplicate-detection/include-exclude-contract'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contract_num: contractNum,
                include: currentlyExcluded // If currently excluded, we want to include it
            })
        })
        .then(response => response.json())
        .then(data => {
            loadingOverlaySpinner.style.display = 'none';
            
            if (data.success) {
                // Update local arrays
                state.includedContracts = data.included_contracts;
                state.excludedContracts = data.excluded_contracts;
                
                const card = document.querySelector(`.contract-card[data-contract-number="${contractNum}"]`);
                if (card) {
                    const isExcluded = state.excludedContracts.includes(contractNum);
                    
                    // Update card class
                    card.className = `contract-card ${isExcluded ? 'excluded' : 'selected'}`;
                    
                    // Update status text
                    const statusText = card.querySelector('.status-text');
                    statusText.className = `status-text ${isExcluded ? 'status-excluded' : 'status-included'}`;
                    statusText.textContent = isExcluded ? 'Excluded' : 'Included';
                    
                    // Update toggle button text
                    const toggleBtn = card.querySelector('.toggle-action-btn');
                    toggleBtn.textContent = isExcluded ? 'Include' : 'Exclude';
                }
                
                // Update included contracts tags
                updateIncludedContractsTags();
            }
        })
        .catch(error => {
            loadingOverlaySpinner.style.display = 'none';
            console.error('Error toggling contract inclusion:', error);
        });
    }

    
    // Filter contracts when searching
    contractSearch.addEventListener('input', function() {
        filterContracts();
    });

    // Also filter when search type changes
    searchTypeSelect.addEventListener('change', function() {
        filterContracts();
    });

    function filterContracts() {
        const searchTerm = contractSearch.value.toLowerCase();
        const searchType = searchTypeSelect.value;
        
        if (!state.allContracts || !state.allContracts.length) return [];
    
        // If search term is empty, show all contracts
        if (!searchTerm.trim()) {
            // Changed this line to display all contracts instead of just returning them
            displayContracts(state.allContracts);
            return state.allContracts;
        }
        
        // Filter contracts by search term and type
        const filteredContracts = state.allContracts.filter(contract => {
            // Check if contract info contains search term
            const contractInfo = [
                contract.contract_number,
                contract.contract_description || '',
                contract.manufacturer_name || ''
            ].join(' ').toLowerCase();
            
            // Apply filter based on search type
            if (searchType === 'contains') {
                return contractInfo.includes(searchTerm);
            } else {
                // 'not-contains'
                return !contractInfo.includes(searchTerm);
            }
        });
        
        displayContracts(filteredContracts);
    }


    // Function to batch toggle all visible contracts
    function batchToggleContracts(include) {
        // Show loading overlay
        loadingOverlaySpinner.style.display = 'flex';
        
        // Get all visible contract cards
        const visibleCards = Array.from(document.querySelectorAll('.contract-card:not([style*="display: none"])'));
        const contractNumbers = visibleCards.map(card => card.dataset.contractNumber);
        
        if (contractNumbers.length === 0) {
            loadingOverlaySpinner.style.display = 'none';
            return;
        }
        
        // Send batch update request to server
        fetch(getApiUrl('/duplicate-detection/batch-update-contracts'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contract_numbers: contractNumbers,
                include: include
            })
        })
        .then(response => response.json())
        .then(data => {
            loadingOverlaySpinner.style.display = 'none';
            
            if (data.success) {
                // Update local arrays
                state.includedContracts = data.included_contracts;
                state.excludedContracts = data.excluded_contracts;
                
                // Update UI for all affected cards
                visibleCards.forEach(card => {
                    const contractNum = card.dataset.contractNumber;
                    const isExcluded = state.excludedContracts.includes(contractNum);
                    
                    // Update card class
                    card.className = `contract-card ${isExcluded ? 'excluded' : 'selected'}`;
                    
                    // Update status text
                    const statusText = card.querySelector('.status-text');
                    statusText.className = `status-text ${isExcluded ? 'status-excluded' : 'status-included'}`;
                    statusText.textContent = isExcluded ? 'Excluded' : 'Included';
                    
                    // Update toggle button text
                    const toggleBtn = card.querySelector('.toggle-action-btn');
                    toggleBtn.textContent = isExcluded ? 'Include' : 'Exclude';
                });
                
                // Update included contracts tags
                updateIncludedContractsTags();
            } else {
                alert('Error updating contracts: ' + data.message);
            }
        })
        .catch(error => {
            loadingOverlaySpinner.style.display = 'none';
            console.error('Error batch updating contracts:', error);
            alert('Error updating contracts: ' + error);
        });
    }
        
    
    function showContractDetails(contractNum) {
        // Show loading
        loadingOverlaySpinner.style.display = 'flex';
        
        fetch(getApiUrl(`/duplicate-detection/get-duplicate-items/${contractNum}`))
        .then(response => response.json())
        .then(data => {
            loadingOverlaySpinner.style.display = 'none';
            
            if (data.success) {
                const contract = data.contract;
                
                // Populate modal with contract info
                document.getElementById('modal-contract-number').textContent = contract.contract_number;
                document.getElementById('modal-contract-description').textContent = contract.contract_description || 'N/A';
                document.getElementById('modal-contract-owner').textContent = contract.contract_owner || 'N/A';
                
                // Populate items table
                const itemsTable = document.getElementById('items-table-modal').querySelector('tbody');
                itemsTable.innerHTML = '';
                
                contract.items.forEach(item => {
                    const row = document.createElement('tr');
                    
                    // Is exact match
                    const isExactMatch = item.same_mfg_part_num === 1;
                    
                    // Add title attribute to show full text on hover for descriptions
                    const ccxDescription = item.description_ccx || 'N/A';
                    const uploadDescription = item.Description || 'N/A';
                    
                    row.innerHTML = `
                        <td>${item.mfg_part_num_ccx || 'N/A'}</td>
                        <td>${item.Mfg_Part_Num || 'N/A'}</td>
                        <td>
                            <span class="modal-match-indicator ${isExactMatch ? 'match' : 'no-match'}">
                                ${isExactMatch ? 'Exact Match' : 'Partial Match'}
                            </span>
                        </td>
                        <td>${item.uom_ccx || 'N/A'}</td>
                        <td>${item.UOM || 'N/A'}</td>
                        <td>$${parseFloat(item.price_ccx).toFixed(2) || 'N/A'}</td>
                        <td>$${parseFloat(item.Contract_Price).toFixed(2) || 'N/A'}</td>
                        <td title="${ccxDescription}">${ccxDescription}</td>
                        <td title="${uploadDescription}">${uploadDescription}</td>
                        <td>${item.contract_number_ccx || 'N/A'}</td>
                        <td>${item.manufacturer_name_ccx || 'N/A'}</td>
                    `;
                    
                    itemsTable.appendChild(row);
                });
                
                // Show modal
                modal.style.display = 'block';
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            loadingOverlaySpinner.style.display = 'none';
            alert('Error: ' + error);
        });
    }
    
    // Close modal when clicking on X button
    closeBtn.addEventListener('click', function() {
        modal.style.display = 'none';
    });
    
    // Close modal when clicking outside of it
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
    
    // this triggers the item level comparison
    // we have SSE call here to get the progress of the item level comparison
    continueBtn.addEventListener('click', function() {
        if (state.includedContracts.length === 0) {
            alert('Please include at least one contract before proceeding.');
            return;
        }
        
        // Show the item-level comparison section
        const itemComparisonContainer = document.getElementById('item-comparison-container');
        if (itemComparisonContainer) {
            itemComparisonContainer.style.display = 'block';
            // Scroll to the item comparison section
            itemComparisonContainer.scrollIntoView({behavior: 'smooth'});
        }

        
        // Show loading overlay with progress bar instead of spinner
        if (loadingOverlayProgress) {
            loadingOverlayProgress.style.display = 'flex';
            
            // Reset progress indicators
            if (progressBar) progressBar.style.width = '0%';
            if (progressPercentage) progressPercentage.textContent = '0%';
            if (progressMessage) progressMessage.textContent = 'Starting comparison...';
            if (itemsProcessed) itemsProcessed.textContent = 'Processed: 0 of 0 items';
        }
        
        // Set up Server-Sent Events for progress updates
        const eventSource = new EventSource('/duplicate-detection/process-item-comparison-with-progress');
        
        eventSource.onmessage = function(event) {
            try {
                console.log("Raw event data:", event.data);
                const data = JSON.parse(event.data);
                console.log("Parsed progress data:", data);
                
                // Update progress indicators
                progressBar.style.width = data.progress + '%';
                progressPercentage.textContent = data.progress + '%';
                progressMessage.textContent = data.message;
                itemsProcessed.textContent = `Processed: ${data.processed} of ${data.total} items`;
                
                // Check if processing is complete
                if (data.status === 'done') {
                    console.log("Processing complete!");
                    eventSource.close();
                    
                    // Call finalize endpoint to ensure session data is saved
                    fetch(getApiUrl('/duplicate-detection/finalize-item-comparison'), {
                        method: 'POST'
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            console.log("Results finalized successfully");                           
                            // Wait a moment before hiding the overlay
                            setTimeout(() => {
                                loadingOverlayProgress.style.display = 'none';                                
                                // Fetch and display summary AFTER finalizing
                                fetchItemComparisonSummary();
                            }, 1000);
                        } else {
                            console.error("Error finalizing results:", data.message);
                            // retry logic for resilience
                            setTimeout(() => {
                                console.log("Retrying to finalize results...");
                                fetch(getApiUrl('/duplicate-detection/finalize-item-comparison'), {
                                    method: 'POST'
                                })
                                .then(response => response.json())
                                .then(retryData => {
                                    if (retryData.success) {
                                        console.log("Results finalized successfully on retry");
                                        loadingOverlayProgress.style.display = 'none';
                                        fetchItemComparisonSummary();
                                    } else {
                                        console.error("Retry Failed:", retryData.message);
                                        alert("Error finalizing results on retry. Please try again.");
                                        loadingOverlayProgress.style.display = 'none';
                                    }
                                })
                                .catch(retryError => {
                                    console.error("Retry Error:", retryError);
                                    alert("Error finalizing results on retry. Please try again.");
                                    loadingOverlayProgress.style.display = 'none';
                                });
                            }, 1000);
                        }
                    })
                    .catch(error => {
                        console.error("Error finalizing results bbb:", error);
                        alert("Error finalizing results. Please try again.");
                        loadingOverlayProgress.style.display = 'none';
                    });
                }
            } catch (error) {
                console.error("Error processing SSE message:", error);
            }
        };
        
        eventSource.onerror = function() {
            eventSource.close();
            progressMessage.textContent = 'Error processing items calling process-item-comparison-with-progress. Please try again.';
            setTimeout(() => {
                loadingOverlayProgress.style.display = 'none';
            }, 3000);
        };
    });

    // Add a new function to make initialization synchronous with callback
    function initializeIncludedContractsWithCallback(callback) {
        console.log("Saving included contracts with callback:", state.includedContracts);
        
        // Send included contracts to the server
        fetch(getApiUrl('/duplicate-detection/initialize-included-contracts'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                contract_numbers: state.includedContracts
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log("Successfully saved included contracts:", data);
                state.includedContracts = data.included_contracts;
                state.excludedContracts = data.excluded_contracts;
                
                // Execute callback after successful initialization
                if (typeof callback === 'function') {
                    callback();
                }
            } else {
                console.error("Error saving included contracts:", data.message);
                alert("Error initializing contracts. Please try again.");
            }
        })
        .catch(error => {
            console.error("Error:", error);
            alert("Error initializing contracts. Please try again.");
        });
    }

    function fetchItemComparisonSummary() {
        console.log("Fetching comparison summary...");
        
        // Check if comparison container exists before proceeding
        if (!comparisonResultsContainer) {
            console.error("comparison-results-container element not found in the DOM");
            alert("UI Error: Could not find results container. Please refresh the page and try again.");
            return;
        }
        
        fetch(getApiUrl('/duplicate-detection/get-item-comparison-summary'))
            .then(response => {
                // Check if response is ok before parsing JSON
                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Got summary data:", data);
                
                if (data.success) {
                    // Make container visible first
                    comparisonResultsContainer.style.display = 'block';
                    
                    // Debug summary data
                    console.log("Summary data structure:", JSON.stringify(data.summary));
                    
                    // Explicitly check if summary data is actually populated
                    if (!data.summary || Object.keys(data.summary).length === 0) {
                        console.error("Summary data is empty or missing");
                        alert("No item comparison data was retrieved. Please try again.");
                        return;
                    }
                    
                    // Update summary counts
                    updateSummaryCounts(data.summary);
                    
                    // Make items detail view visible
                    itemsDetailView.style.display = 'block';
                    
                    // Load appropriate confidence level items
                    if (data.summary.high && data.summary.high.total > 0) {
                        console.log("Loading high confidence items...");
                        loadConfidenceItems('high');
                    } else if (data.summary.medium && data.summary.medium.total > 0) {
                        console.log("Loading medium confidence items...");
                        loadConfidenceItems('medium');
                    } else if (data.summary.low && data.summary.low.total > 0) {
                        console.log("Loading low confidence items...");
                        loadConfidenceItems('low');
                    } else {
                        console.warn("No items found in any confidence level");
                        noMatchesMessage.style.display = 'block';
                        document.querySelector('.items-table-container').style.display = 'none';
                    }
                } else {
                    console.error("Error in summary data:", data.message);
                    alert("Error loading comparison results: " + data.message);
                }
            })
            .catch(error => {
                console.error("Error fetching summary:", error);
                // Show more informative error message for users
                alert("Error loading comparison results. Please check the console for details and try again.");
                // Hide the progress overlay
                loadingOverlayProgress.style.display = 'none';
            });
    }

    // Confidence card click handlers
    document.querySelectorAll('.confidence-card').forEach(card => {
        card.addEventListener('click', function() {
            const level = this.dataset.level;
            loadConfidenceItems(level);
        });
    });
    
    // Table header click for sorting
    document.querySelectorAll('.items-table th[data-sort]').forEach(th => {
        th.addEventListener('click', function() {
            const field = this.dataset.sort;
            
            // Toggle direction if same field, otherwise default to asc
            if (field === sortField) {
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                sortField = field;
                sortDirection = 'asc';
            }
            
            // Update sort indicators
            document.querySelectorAll('.items-table th').forEach(header => {
                header.classList.remove('sort-asc', 'sort-desc');
            });
            
            this.classList.add(sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
            
            // Re-render the table with new sort
            renderItemsTable(currentItems);
        });
    });
    
    // Filter input handler
    // Add event listeners for filtering
    itemSearchInput.addEventListener('input', filterItemTable);
    itemSearchType.addEventListener('change', filterItemTable);

    function filterItemTable() {
        const searchTerm = itemSearchInput.value.toLowerCase();
        const searchType = itemSearchType.value;
        const tableRows = document.querySelectorAll('#items-tbody tr');
        
        tableRows.forEach(row => {
            const text = row.textContent.toLowerCase();
            const matchesContains = text.includes(searchTerm);
            
            // Show/hide based on search type
            if (searchType === 'contains') {
                row.style.display = matchesContains ? '' : 'none';
            } else { // not-contains
                row.style.display = matchesContains ? 'none' : '';
            }
        });
        
        // Update visible count
        updateVisibleItemCount();
    }

    clearSearchBtn.addEventListener('click', function() {
        // Clear the search input
        itemSearchInput.value = '';
        // Reset search type to "contains"
        itemSearchType.value = 'contains';
        // Trigger filtering to show all items
        filterItemTable();
    });

    function updateVisibleItemCount() {
        const visibleRows = document.querySelectorAll('#items-tbody tr[style="display: none;"]').length;
        const totalRows = document.querySelectorAll('#items-tbody tr').length;
        const visibleCount = totalRows - visibleRows;
        
        // If you have an element to display the count, update it here
        const countElement = document.getElementById('visible-item-count');
        if (countElement) {
            countElement.textContent = `Showing ${visibleCount} of ${totalRows} items`;
        }
    }
    
    // Select/Deselect all buttons
    selectAllBtn.addEventListener('click', function() {
        document.querySelectorAll('#items-tbody tr:not([style*="display: none"]) input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = true;
        });
    });
    
    deselectAllBtn.addEventListener('click', function() {
        document.querySelectorAll('#items-tbody tr:not([style*="display: none"]) input[type="checkbox"]').forEach(checkbox => {
            checkbox.checked = false;
        });
    });
    
    // Save selections button
    saveSelectionsBtn.addEventListener('click', function() {
        saveSelections();
    });
    
    // Function to update summary counts
    function updateSummaryCounts(summary) {
        document.getElementById('high-count').textContent = summary.high.total;
        document.getElementById('medium-count').textContent = summary.medium.total;
        document.getElementById('low-count').textContent = summary.low.total;
        
        document.getElementById('high-false-positive-count').textContent = 
            summary.high.false_positives + ' marked as false positive';
        document.getElementById('medium-false-positive-count').textContent = 
            summary.medium.false_positives + ' marked as false positive';
        document.getElementById('low-false-positive-count').textContent = 
            summary.low.false_positives + ' marked as false positive';
    }
    
    // Function to load items for a confidence level
    function loadConfidenceItems(level) {
        // Update selected card
        document.querySelectorAll('.confidence-card').forEach(card => {
            card.classList.remove('selected');
        });
        document.getElementById(level + '-confidence-card').classList.add('selected');
        
        // Update level title
        detailTitle.textContent = level.charAt(0).toUpperCase() + level.slice(1) + ' Confidence Matches';
        
        loadingOverlaySpinner.style.display = 'flex';
        currentLevel = level;
        
        console.log(`Fetching items for confidence level: ${level}`);
        
        fetch(getApiUrl(`/duplicate-detection/get-comparison-items/${level}`))
            .then(response => response.json())
            .then(data => {
                loadingOverlaySpinner.style.display = 'none';
                
                if (data.success) {
                    console.log(`Received ${data.count} items for ${level} level`);
                    
                    // Store current items
                    currentItems = data.items;
                    
                    // Reset sorting and filtering
                    sortField = null;
                    sortDirection = 'asc';
                    document.querySelectorAll('.items-table th').forEach(header => {
                        header.classList.remove('sort-asc', 'sort-desc');
                    });
                    
                    // Render items
                    renderItemsTable(currentItems);
                    
                    // Show the detail view - this should already be visible from fetchItemComparisonSummary
                    // but ensure it's still visible
                    itemsDetailView.style.display = 'block';
                    
                    // Show/hide no matches message
                    noMatchesMessage.style.display = currentItems.length === 0 ? 'block' : 'none';
                    document.querySelector('.items-table-container').style.display = 
                        currentItems.length === 0 ? 'none' : 'block';

                    // Reapply filter after rendering
                    filterItemTable();
                } else {
                    console.error(`Error fetching ${level} items:`, data.message);
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                loadingOverlaySpinner.style.display = 'none';
                console.error(`Error in fetch for ${level} items:`, error);
                alert('Error: ' + error);
            });
    }
    
    // Function to render the items table
    function renderItemsTable(items) {
        // Clear the table
        itemsTbody.innerHTML = '';
        
        // Sort items if sort field is set
        if (sortField) {
            items = [...items].sort((a, b) => {
                let valA = a[sortField];
                let valB = b[sortField];
                
                // Handle numeric fields
                if (typeof valA === 'number' && typeof valB === 'number') {
                    return sortDirection === 'asc' ? valA - valB : valB - valA;
                }
                
                // Handle string fields
                valA = String(valA || '').toLowerCase();
                valB = String(valB || '').toLowerCase();
                
                if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
                if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
                return 0;
            });
        }
        
        // Add rows to table
        items.forEach((item, index) => {
            const row = document.createElement('tr');
            
            // Get confidence class based on score
            const scoreClass = item.weighted_score >= 0.8 ? 'high-score' : 
                            (item.weighted_score >= 0.6 ? 'medium-score' : 'low-score');
            
            // Format EA prices
            const ccxEaPrice = item.ccx_ea_price !== null ? 
                '$' + parseFloat(item.ccx_ea_price).toFixed(2) : 'N/A';
            const uploadEaPrice = item.upload_ea_price !== null ? 
                '$' + parseFloat(item.upload_ea_price).toFixed(2) : 'N/A';
            
            row.innerHTML = `
                <td>${item.mfg_part_num_ccx || 'N/A'}</td>
                <td>${item.Mfg_Part_Num || 'N/A'}</td>
                <td>${item.uom_ccx || 'N/A'}</td>
                <td>${item.UOM || 'N/A'}</td>
                <td>${item.qoe_ccx || 'N/A'}</td>
                <td>${item.QOE || 'N/A'}</td>
                <td>${ccxEaPrice}</td>
                <td>${uploadEaPrice}</td>
                <td>
                    <div class="match-indicator ${item.same_mfg_part_num === 1 ? 'match' : 'no-match'}"></div>
                </td>
                <td class="score-cell ${scoreClass}">${Math.round(item.weighted_score * 100)}%</td>
                <td>
                    <input type="checkbox" class="false-positive-checkbox" 
                        data-index="${index}" 
                        ${item.false_positive ? 'checked' : ''}>
                </td>
                <td title="${item.description_ccx || 'N/A'}">${(item.description_ccx || 'N/A').substring(0, 50)}${(item.description_ccx && item.description_ccx.length > 50) ? '...' : ''}</td>
                <td title="${item.Description || 'N/A'}">${(item.Description || 'N/A').substring(0, 50)}${(item.Description && item.Description.length > 50) ? '...' : ''}</td>
                <td>${item.contract_number_ccx || 'N/A'}</td>
                <td>${item.manufacturer_name_ccx || 'N/A'}</td>
                <td>${item.File_Row || 'N/A'}</td>
            `;
            
            itemsTbody.appendChild(row);
        });
    }
    
    
    // Function to save false positive selections
    function saveSelections() {
        // Show loading overlay
        loadingOverlaySpinner.style.display = 'flex';
        
        // Get all checkboxes and their status
        const checkboxes = document.querySelectorAll('#items-tbody input[type="checkbox"]');
        
        // Create a map of all items and their status
        const itemUpdates = Array.from(checkboxes).map(checkbox => {
            return {
                index: parseInt(checkbox.dataset.index),
                is_false_positive: checkbox.checked
            };
        });
        
        // Send a single request with complete state
        fetch(getApiUrl('/duplicate-detection/update-false-positives'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                confidence_level: currentLevel,
                item_updates: itemUpdates
            })
        })
        .then(response => response.json())
        .then(data => {
            loadingOverlaySpinner.style.display = 'none';
            
            if (data.success) {
                // Update the item states in the local array
                itemUpdates.forEach(update => {
                    if (update.index < currentItems.length) {
                        currentItems[update.index].false_positive = update.is_false_positive;
                    }
                });
                
                // Update the summary display
                updateSummaryCounts(data.summary);
                
                // Show success message
                alert('Selections saved successfully');
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            loadingOverlaySpinner.style.display = 'none';
            alert('Error saving selections: ' + error.message);
        });
    }
});