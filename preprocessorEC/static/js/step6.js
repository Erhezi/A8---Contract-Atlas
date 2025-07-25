document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const previewBtn = document.getElementById('preview-btn');
    const exportAllBtn = document.getElementById('export-all-btn');
    const loadingSpinner = document.getElementById('loading-spinner');
    const skipGpoCheckbox = document.getElementById('skip_gpo');
    const exportFormatSelect = document.getElementById('export_format');
    const batchContractFilterSelect = document.getElementById('batch-contract-filter');
    const contractFilterSelect = document.getElementById('contract-filter');

    const fileLinksContainer = document.getElementById('file-links-container');
    const itemLinkPreview = document.getElementById('item-link-preview');
    const contractToClosePreview = document.getElementById('contract-to-close-preview');
    
    // Pagination constants
    const ROWS_PER_PAGE = 1000;
    
    // Pagination state variables
    const paginationState = {
        batch: { currentPage: 1, totalPages: 1, allData: [], filteredData: [] },
        single: { currentPage: 1, totalPages: 1, allData: [], filteredData: [] },
        itemLink: { allData: [], filteredData: [] }, // Removed pagination properties
        contractToClose: { allData: [], filteredData: [] } // Removed pagination properties
    };
    
    // Pagination controls
    const paginationControls = {
        batch: {
            prevButton: document.getElementById('batch-prev-page'),
            nextButton: document.getElementById('batch-next-page'),
            currentPageEl: document.getElementById('batch-current-page'),
            totalPagesEl: document.getElementById('batch-total-pages')
        },
        single: {
            prevButton: document.getElementById('single-prev-page'),
            nextButton: document.getElementById('single-next-page'),
            currentPageEl: document.getElementById('single-current-page'),
            totalPagesEl: document.getElementById('single-total-pages')
        }
        // Removed itemLink and contractToClose pagination controls
    };
    
    // Setup pagination event listeners
    if (paginationControls.batch.prevButton) {
        paginationControls.batch.prevButton.addEventListener('click', () => {
            if (paginationState.batch.currentPage > 1) {
                paginationState.batch.currentPage--;
                renderBatchTablePage();
            }
        });
    }
    
    if (paginationControls.batch.nextButton) {
        paginationControls.batch.nextButton.addEventListener('click', () => {
            if (paginationState.batch.currentPage < paginationState.batch.totalPages) {
                paginationState.batch.currentPage++;
                renderBatchTablePage();
            }
        });
    }
    
    if (paginationControls.single.prevButton) {
        paginationControls.single.prevButton.addEventListener('click', () => {
            if (paginationState.single.currentPage > 1) {
                paginationState.single.currentPage--;
                renderSingleTablePage();
            }
        });
    }
    
    if (paginationControls.single.nextButton) {
        paginationControls.single.nextButton.addEventListener('click', () => {
            if (paginationState.single.currentPage < paginationState.single.totalPages) {
                paginationState.single.currentPage++;
                renderSingleTablePage();
            }
        });
    }
    
    // Removed pagination event listeners for itemLink and contractToClose tables

    // If batch contract filter exists, add event listener
    if (batchContractFilterSelect) {
        batchContractFilterSelect.addEventListener('change', function() {
            const selectedContract = this.value;
            filterBatchTableByContract(selectedContract);
        });
    }

    // If contract filter exists, add event listener
    if (contractFilterSelect) {
        contractFilterSelect.addEventListener('change', function() {
            const selectedContract = this.value;
            filterTableByContract(selectedContract);
        });
    }

    const userRole = document.getElementById('user-role')?.value || '';
    const isSourcingUser = userRole === 'sourcing';
    
    // Cache DOM elements
    const completeStepContainer = document.getElementById('complete-step-container');
    

    // Preview data function
    if (previewBtn) {
        previewBtn.addEventListener('click', function() {
            console.log('Preview button clicked');

            if (loadingSpinner) {
                loadingSpinner.style.display = 'flex';
            }

            // Prepare data for the request
            const requestData = {
                skip_gpo: skipGpoCheckbox.checked,
                export_format: exportFormatSelect.value
            };

            // Fetch data preview
            fetch(getApiUrl('/data-export/preview-data'), {
                method: 'POST',
                body: JSON.stringify(requestData),
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                console.log('Response status:', response.status);
                return response.json();
            })
            .then(data => {
                loadingSpinner.style.display = 'none';

                if (data.success) {

                    // Show the preview section
                    const dataPreview = document.getElementById('data-preview');
                    if (dataPreview) {
                        dataPreview.style.display = 'block';
                    }

                    // Show the horizontal line
                    const horizontalLine = document.getElementById('horizontal-line');
                    if (horizontalLine) {
                        horizontalLine.style.display = 'block';
                    }

                    // show the item and contract to close previews
                    if (itemLinkPreview) {
                        itemLinkPreview.style.display = 'block';
                    }

                    if (contractToClosePreview) {
                        contractToClosePreview.style.display = 'block';
                    }

                    // Show the export button
                    if (exportAllBtn) {
                        exportAllBtn.style.display = 'block';
                    }

                    // Reset pagination state for batch and single tables
                    paginationState.batch.currentPage = 1;
                    paginationState.single.currentPage = 1;

                    // Store all data in pagination state
                    paginationState.batch.allData = data.batch_data || [];
                    paginationState.batch.filteredData = [...paginationState.batch.allData];
                    
                    paginationState.single.allData = data.single_data || [];
                    paginationState.single.filteredData = [...paginationState.single.allData];
                    
                    paginationState.itemLink.allData = data.item_link_data || [];
                    paginationState.itemLink.filteredData = [...paginationState.itemLink.allData];
                    
                    paginationState.contractToClose.allData = data.contract_to_close_data || [];
                    paginationState.contractToClose.filteredData = [...paginationState.contractToClose.allData];

                    // Update tables
                    updateBatchTable(paginationState.batch.allData);
                    updateSingleTable(paginationState.single.allData);
                    updateItemLinkTable(paginationState.itemLink.allData);
                    updateContractToCloseTable(paginationState.contractToClose.allData);

                
                    // Enable export button if there's data
                    if (exportAllBtn && (
                        paginationState.batch.allData.length > 0 || 
                        paginationState.single.allData.length > 0 ||
                        paginationState.itemLink.allData.length > 0 ||
                        paginationState.contractToClose.allData.length > 0
                    )) {
                        exportAllBtn.disabled = false;
                        exportAllBtn.classList.remove('committed'); // Remove committed class if it was previously added
                        exportAllBtn.innerHTML = '<i class="fas fa-file-export"></i>Export for Execution';
                    } else if (exportAllBtn) {
                        exportAllBtn.disabled = true;
                        exportAllBtn.classList.add('committed'); // Add committed class for better visual indication
                        exportAllBtn.innerHTML = '<i class="fas fa-ban mr-2"></i>No Data Available';
                        // Show the appropriate step complete button if no data
                        const completeStepContainer = document.getElementById('complete-step-container');
                        if (completeStepContainer) {
                            completeStepContainer.style.display = 'block';
                        }
                    }

                    // Always show the complete step button for sourcing users after preview
                    if (isSourcingUser && completeStepContainer) {
                        completeStepContainer.style.display = 'block';
                    }

                    // Scroll to the preview section
                    dataPreview.scrollIntoView({ behavior: 'smooth' });
                } else {
                    alert('Error loading preview data: ' + data.message);
                }
            })
            .catch(error => {
                loadingSpinner.style.display = 'none';
                console.error('Error:', error);
                alert('An error occurred while loading preview data. Please try again.');
            });
        });
    }

    // Add event listener for export button
    if (exportAllBtn) {
        exportAllBtn.addEventListener('click', function() {
            // Disable the button to prevent multiple clicks
            exportAllBtn.disabled = true;
            exportAllBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Exporting...';
            
            // Show loading spinner
            if (loadingSpinner) {
                loadingSpinner.style.display = 'flex';
            }

            // Prepare data for the request
            const requestData = {
                skip_gpo: skipGpoCheckbox.checked,
                export_format: exportFormatSelect.value
            };

            // Send export request
            fetch(getApiUrl('/data-export/export-data'), {
                method: 'POST',
                body: JSON.stringify(requestData),
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading spinner
                loadingSpinner.style.display = 'none';
                
                if (data.success) {
                    // Clear existing file links
                    if (fileLinksContainer) {
                        fileLinksContainer.innerHTML = '';
                    }
                    
                    // Add the zip file link
                    if (data.zipFile) {
                        // make contianer visible
                        fileLinksContainer.style.display = 'block';

                        const fileLink = document.createElement('a');
                        fileLink.href = data.zipFile.url;
                        fileLink.className = 'btn btn-outline-primary file-link';
                        fileLink.innerHTML = `<i class="fas fa-file-archive mr-1"></i> ${data.zipFile.name}`;
                        fileLink.setAttribute('download', data.zipFile.name);
                        fileLinksContainer.appendChild(fileLink);
                    }
                    
                    // Show success message
                    alert('Export completed successfully! Click the link to download your files.');
                    
                    // Mark button as committed
                    exportAllBtn.classList.add('committed');
                    exportAllBtn.disabled = true;
                    exportAllBtn.innerHTML = '<i class="fas fa-check mr-2"></i>Changes Exported';
                    
                    // Show complete step container
                    if (completeStepContainer) {
                        completeStepContainer.style.display = 'block';
                    }
                } else {
                    // Show error message
                    alert('Export failed: ' + data.message);
                    
                    // Reset button
                    exportAllBtn.disabled = false;
                    exportAllBtn.innerHTML = '<i class="fas fa-file-export"></i>Export for Execution';
                }
            })
            .catch(error => {
                // Hide loading spinner
                loadingSpinner.style.display = 'none';
                
                // Show error message
                console.error('Error:', error);
                alert('An error occurred during export. Please try again.');
                
                // Reset button
                exportAllBtn.disabled = false;
                exportAllBtn.innerHTML = '<i class="fas fa-file-export"></i>Export for Execution';
            });
        });
    }

    if (isSourcingUser && completeStepContainer) {
        // Check if data is already loaded
        const dataPreview = document.getElementById('data-preview');
        if (dataPreview && window.getComputedStyle(dataPreview).display !== 'none') {
            completeStepContainer.style.display = 'block';
        }
    }

    // function to do filter batch table by contract number
    function filterBatchTableByContract(contractNumber) {
        const batchTable = document.getElementById('batch-table');
        if (!batchTable) return;
        
        // Reset to page 1 when filtering
        paginationState.batch.currentPage = 1;
        
        // If "all" is selected, show all rows
        if (contractNumber === 'all') {
            paginationState.batch.filteredData = [...paginationState.batch.allData];
        } else {
            // Filter the data based on contract number
            paginationState.batch.filteredData = paginationState.batch.allData.filter(record => 
                record['Contract Number'] === contractNumber
            );
        }
        
        // Update pagination and render
        updateBatchPagination();
        renderBatchTablePage();
    }

    // Update pagination info for batch table
    function updateBatchPagination() {
        const totalItems = paginationState.batch.filteredData.length;
        paginationState.batch.totalPages = Math.max(1, Math.ceil(totalItems / ROWS_PER_PAGE));
        
        // Update pagination controls
        if (paginationControls.batch.totalPagesEl) {
            paginationControls.batch.totalPagesEl.textContent = paginationState.batch.totalPages;
        }
        if (paginationControls.batch.currentPageEl) {
            paginationControls.batch.currentPageEl.textContent = paginationState.batch.currentPage;
        }
        
        // Update button states
        if (paginationControls.batch.prevButton) {
            paginationControls.batch.prevButton.disabled = paginationState.batch.currentPage <= 1;
        }
        if (paginationControls.batch.nextButton) {
            paginationControls.batch.nextButton.disabled = 
                paginationState.batch.currentPage >= paginationState.batch.totalPages;
        }
        
        // Update count badge
        const batchCount = document.getElementById('batch-count');
        if (batchCount) {
            batchCount.textContent = `${totalItems} records`;
        }
    }
    
    // Render current page of batch table
    function renderBatchTablePage() {
        const batchTable = document.getElementById('batch-table');
        const batchNoData = document.getElementById('batch-no-data');
        
        if (!batchTable) return;
        
        const tbody = batchTable.querySelector('tbody');
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        // Calculate start and end indices for current page
        const startIdx = (paginationState.batch.currentPage - 1) * ROWS_PER_PAGE;
        const endIdx = Math.min(startIdx + ROWS_PER_PAGE, paginationState.batch.filteredData.length);
        
        // Get current page data
        const currentPageData = paginationState.batch.filteredData.slice(startIdx, endIdx);
        
        // Update pagination controls
        updateBatchPagination();
        
        if (currentPageData.length === 0) {
            // Show no data message
            if (batchNoData) batchNoData.style.display = 'block';
            
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="21" class="no-data-message">No records to display</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message and populate table
            if (batchNoData) batchNoData.style.display = 'none';
            
            currentPageData.forEach(record => {
                const row = document.createElement('tr');

                // Check for date conflicts
                const hasDateConflict = 
                    (record['Final Date L Check'] && record['Final Date L Check'] !== 'pass') || 
                    (record['Final Date H Check'] && record['Final Date H Check'] !== 'pass');

                // Check for missing vendor part with additional condition
                const missingVendorPart = !record['Vendor Part Num'] && 
                    ['Create', 'Expire then Create (Create)', 'Update (New)'].includes(record['Actual Action']);
                
                // Check for possible duplicates (Create Count > 1)
                const possibleDuplicate = record['Create Count'] && parseInt(record['Create Count']) > 1;

                // Apply appropriate class (prioritize date conflict over possible duplicate over missing vendor part)
                if (hasDateConflict) {
                    row.classList.add('date-conflict');
                } else if (possibleDuplicate) {
                    row.classList.add('possible-duplicate');
                } else if (missingVendorPart) {
                    row.classList.add('missing-vendor-part');
                }

                // Add the row HTML
                row.innerHTML = `
                    <td title="${escapeHtml(record.Organization || '')}">${escapeHtml(record.Organization || '')}</td>
                    <td title="${escapeHtml(record.Vendor || '')}">${escapeHtml(record.Vendor || '')}</td>
                    <td title="${escapeHtml(record.Manufacturer || '')}">${escapeHtml(record.Manufacturer || '')}</td>
                    <td title="${escapeHtml(record['Contract Number'] || '')}">${escapeHtml(record['Contract Number'] || '')}</td>
                    <td title="${escapeHtml(record['Contract Description'] || '')}">${escapeHtml(record['Contract Description'] || '')}</td>
                    <td title="${escapeHtml(record['Tier Level'] || '')}">${escapeHtml(record['Tier Level'] || '')}</td>
                    <td title="${escapeHtml(record['Tier Description'] || '')}">${escapeHtml(record['Tier Description'] || '')}</td>
                    <td title="${escapeHtml(record['Source Type'] || '')}">${escapeHtml(record['Source Type'] || '')}</td>
                    <td title="${formatDate(record['Start Date'])}">${formatDate(record['Start Date'])}</td>
                    <td title="${formatDate(record['End Date'])}">${formatDate(record['End Date'])}</td>
                    <td title="${escapeHtml(record['Mfg Part Num'] || '')}">${escapeHtml(record['Mfg Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Vendor Part Num'] || '')}">${escapeHtml(record['Vendor Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Buyer Part Num'] || '')}">${escapeHtml(record['Buyer Part Num'] || '')}</td>
                    <td class="description-col" title="${escapeHtml(record.Description || '')}">${escapeHtml(record.Description || '')}</td>
                    <td title="${record['Contract Price'] || ''}">${record['Contract Price'] || ''}</td>
                    <td title="${escapeHtml(record.UOM || '')}">${escapeHtml(record.UOM || '')}</td>
                    <td title="${record.QOE || ''}">${record.QOE || ''}</td>
                    <td title="${formatDate(record['Effective Date'])}">${formatDate(record['Effective Date'])}</td>
                    <td title="${formatDate(record['Expiration Date'])}">${formatDate(record['Expiration Date'])}</td>
                    <td class="non-export-col" title="${escapeHtml(record['Actual Action'] || '')}">${escapeHtml(record['Actual Action'] || '')}</td>
                    <td class="non-export-col" title="${escapeHtml(record['ERP Vendor ID (CCX Sync)'] || '')}">${escapeHtml(record['ERP Vendor ID (CCX Sync)'] || '')}</td>
                `;
                tbody.appendChild(row);
            });
        }
    }

    // Function to update batch table with data
    function updateBatchTable(batchData) {
        const batchTable = document.getElementById('batch-table');
        const batchNoData = document.getElementById('batch-no-data');

        if (!batchTable) {
            console.error('Batch table elements not found');
            return;
        }
        
        // Store the data
        paginationState.batch.allData = batchData;
        paginationState.batch.filteredData = [...batchData];
        paginationState.batch.currentPage = 1;
        
        if (batchData.length === 0) {
            // Show no data message
            if (batchNoData) batchNoData.style.display = 'block';

            // Disable and clear batch contract filter
            if (batchContractFilterSelect) {
                batchContractFilterSelect.innerHTML = '<option value="all">All Contracts</option>';
                batchContractFilterSelect.disabled = true;
            }

            const explanationDiv = document.querySelector('.card-body .highlight-explanation');
            if (explanationDiv) {
                explanationDiv.style.display = 'none';
            }
        } else {
            // Populate batch contract filter dropdown
            if (batchContractFilterSelect) {
                // Get unique contract numbers
                const uniqueContracts = [...new Set(batchData.map(record => record['Contract Number']))];

                // Clear previous options except "All Contracts"
                batchContractFilterSelect.innerHTML = '<option value="all">All Contracts</option>';

                // Add each contract as an option
                uniqueContracts.forEach(contract => {
                    if (contract) { // Skip empty values
                        const option = document.createElement('option');
                        option.value = contract;
                        option.textContent = contract;
                        batchContractFilterSelect.appendChild(option);
                    }
                });

                // Enable the dropdown
                batchContractFilterSelect.disabled = false;
            }
        }
        
        // Update pagination and render first page
        updateBatchPagination();
        renderBatchTablePage();
    }

    // function to do filter single table by contract number
    function filterTableByContract(contractNumber) {
        const singleTable = document.getElementById('single-table');
        if (!singleTable) return;
        
        // Reset to page 1 when filtering
        paginationState.single.currentPage = 1;
        
        // If "all" is selected, show all rows
        if (contractNumber === 'all') {
            paginationState.single.filteredData = [...paginationState.single.allData];
        } else {
            // Filter the data based on contract number
            paginationState.single.filteredData = paginationState.single.allData.filter(record => 
                record['Contract Number (PrP)'] === contractNumber
            );
        }
        
        // Update pagination and render
        updateSinglePagination();
        renderSingleTablePage();
    }

    // Update pagination info for single table
    function updateSinglePagination() {
        const totalItems = paginationState.single.filteredData.length;
        paginationState.single.totalPages = Math.max(1, Math.ceil(totalItems / ROWS_PER_PAGE));
        
        // Update pagination controls
        if (paginationControls.single.totalPagesEl) {
            paginationControls.single.totalPagesEl.textContent = paginationState.single.totalPages;
        }
        if (paginationControls.single.currentPageEl) {
            paginationControls.single.currentPageEl.textContent = paginationState.single.currentPage;
        }
        
        // Update button states
        if (paginationControls.single.prevButton) {
            paginationControls.single.prevButton.disabled = paginationState.single.currentPage <= 1;
        }
        if (paginationControls.single.nextButton) {
            paginationControls.single.nextButton.disabled = 
                paginationState.single.currentPage >= paginationState.single.totalPages;
        }
        
        // Update count badge
        const singleCount = document.getElementById('single-count');
        if (singleCount) {
            singleCount.textContent = `${totalItems} records`;
        }
    }
    
    // Render current page of single table
    function renderSingleTablePage() {
        const singleTable = document.getElementById('single-table');
        const singleNoData = document.getElementById('single-no-data');
        
        if (!singleTable) return;
        
        const tbody = singleTable.querySelector('tbody');
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        // Calculate start and end indices for current page
        const startIdx = (paginationState.single.currentPage - 1) * ROWS_PER_PAGE;
        const endIdx = Math.min(startIdx + ROWS_PER_PAGE, paginationState.single.filteredData.length);
        
        // Get current page data
        const currentPageData = paginationState.single.filteredData.slice(startIdx, endIdx);
        
        // Update pagination controls
        updateSinglePagination();
        
        if (currentPageData.length === 0) {
            // Show no data message
            if (singleNoData) singleNoData.style.display = 'block';
            
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="11" class="no-data-message">No records to display</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message and populate table
            if (singleNoData) singleNoData.style.display = 'none';
            
            currentPageData.forEach(record => {
                const row = document.createElement('tr');

                // Check for date conflicts
                const hasDateConflict = 
                    (record['Final Date L Check'] && record['Final Date L Check'] !== 'pass') || 
                    (record['Final Date H Check'] && record['Final Date H Check'] !== 'pass');

                // Check for missing vendor part with additional condition
                const missingVendorPart = !record['Vendor Part Num'] && 
                    ['Create', 'Expire then Create (Create)', 'Update (New)'].includes(record['Actual Action']);
                
                // Check for possible duplicates (Create Count > 1)
                const possibleDuplicate = record['Create Count'] && parseInt(record['Create Count']) > 1;

                // Apply appropriate class (prioritize date conflict over possible duplicate over missing vendor part)
                if (hasDateConflict) {
                    row.classList.add('date-conflict');
                } else if (possibleDuplicate) {
                    row.classList.add('possible-duplicate');
                } else if (missingVendorPart) {
                    row.classList.add('missing-vendor-part');
                }

                row.innerHTML = `
                    <td title="${escapeHtml(record['Mfg Part Num'] || '')}">${escapeHtml(record['Mfg Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Vendor Part Num'] || '')}">${escapeHtml(record['Vendor Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Buyer Part Num'] || '')}">${escapeHtml(record['Buyer Part Num'] || '')}</td>
                    <td class="description-col" title="${escapeHtml(record.Description || '')}">${escapeHtml(record.Description || '')}</td>
                    <td title="${record['Contract Price'] || ''}">${record['Contract Price'] || ''}</td>
                    <td title="${escapeHtml(record.UOM || '')}">${escapeHtml(record.UOM || '')}</td>
                    <td title="${record.QOE || ''}">${record.QOE || ''}</td>
                    <td title="${formatDate(record['Effective Date'])}">${formatDate(record['Effective Date'])}</td>
                    <td title="${formatDate(record['Expiration Date'])}">${formatDate(record['Expiration Date'])}</td>
                    <td class="non-export-col" title="${escapeHtml(record['Contract Number (PrP)'] || '')}">${escapeHtml(record['Contract Number (PrP)'] || '')}</td>
                    <td class="non-export-col" title="${escapeHtml(record['ERP Vendor ID (PrP)'] || '')}">${escapeHtml(record['ERP Vendor ID (PrP)'] || '')}</td>
                `;
                tbody.appendChild(row);
            });
        }
    }

    // Function to update single table with data
    function updateSingleTable(singleData) {
        const singleTable = document.getElementById('single-table');
        const singleNoData = document.getElementById('single-no-data');
        const contractFilterSelect = document.getElementById('contract-filter');

        if (!singleTable) {
            console.error('Single table elements not found');
            return;
        }
        
        // Store the data
        paginationState.single.allData = singleData;
        paginationState.single.filteredData = [...singleData];
        paginationState.single.currentPage = 1;
        
        if (singleData.length === 0) {
            // Show no data message
            if (singleNoData) singleNoData.style.display = 'block';

            // Disable and clear contract filter
            if (contractFilterSelect) {
                contractFilterSelect.innerHTML = '<option value="all">All Contracts</option>';
                contractFilterSelect.disabled = true;
            }

            const explanationDiv = document.querySelector('.card-body .highlight-explanation');
            if (explanationDiv) {
                explanationDiv.style.display = 'none';
            }
        } else {
            // Populate contract filter dropdown
            if (contractFilterSelect) {
                // Get unique contract numbers
                const uniqueContracts = [...new Set(singleData.map(record => record['Contract Number (PrP)']))];

                // Clear previous options except "All Contracts"
                contractFilterSelect.innerHTML = '<option value="all">All Contracts</option>';

                // Add each contract as an option
                uniqueContracts.forEach(contract => {
                    if (contract) { // Skip empty values
                        const option = document.createElement('option');
                        option.value = contract;
                        option.textContent = contract;
                        contractFilterSelect.appendChild(option);
                    }
                });

                // Enable the dropdown
                contractFilterSelect.disabled = false;
            }
        }
        
        // Update pagination and render first page
        updateSinglePagination();
        renderSingleTablePage();
    }

    // Function to update item link table
    function updateItemLinkTable(itemLinkData) {
        const itemLinkTable = document.getElementById('item-link-table');
        const itemLinkNoData = document.getElementById('item-link-no-data');
        
        if (!itemLinkTable) {
            console.error('Item link table elements not found');
            return;
        }
        
        // Store the data
        paginationState.itemLink.allData = itemLinkData;
        paginationState.itemLink.filteredData = [...itemLinkData];
        
        // Get table body
        const tbody = itemLinkTable.querySelector('tbody');
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        // Update count badge - simplified to just show total records
        const itemLinkCount = document.getElementById('item-link-count');
        if (itemLinkCount) {
            itemLinkCount.textContent = `${itemLinkData.length} records`;
        }
        
        if (itemLinkData.length === 0) {
            // Show no data message
            if (itemLinkNoData) itemLinkNoData.style.display = 'block';
            
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="14" class="no-data-message">No records to display</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message and populate table with all data
            if (itemLinkNoData) itemLinkNoData.style.display = 'none';
            
            itemLinkData.forEach(record => {
                const row = document.createElement('tr');
                
                // Check for conditions that require red highlighting
                const hasError = 
                    (record['Inconsistent Mfg Part Num'] === 'Inconsistent') || 
                    (record['Invalid Buy UOM'] === 'Invalid') || 
                    (record['Invalid Item'] === 'Invalid');
                
                // Check for condition that requires yellow highlighting
                const isManualLink = record['Item Master Auto Link'] === 'Manual';
                
                // Apply appropriate class (prioritize red over yellow)
                if (hasError) {
                    row.classList.add('item-error'); // Red highlighting
                } else if (isManualLink) {
                    row.classList.add('manual-link'); // Yellow highlighting
                }
                
                // Add the row HTML
                row.innerHTML = `
                    <td title="${escapeHtml(record['Contract Number'] || '')}">${escapeHtml(record['Contract Number'] || '')}</td>
                    <td title="${escapeHtml(record['Vendor'] || '')}">${escapeHtml(record['Vendor'] || '')}</td>
                    <td title="${escapeHtml(record['VendorItem'] || '')}">${escapeHtml(record['VendorItem'] || '')}</td>
                    <td title="${escapeHtml(record['UOM'] || '')}">${escapeHtml(record['UOM'] || '')}</td>
                    <td title="${record['QOE'] || ''}">${record['QOE'] || ''}</td>
                    <td class="description-col" title="${escapeHtml(record['Description'] || '')}">${escapeHtml(record['Description'] || '')}</td>
                    <td title="${escapeHtml(record['Mfg Part Num'] || '')}">${escapeHtml(record['Mfg Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Item'] || '')}">${escapeHtml(record['Item'] || '')}</td>
                    <td title="${escapeHtml(record['Item Master Auto Link'] || '')}">${escapeHtml(record['Item Master Auto Link'] || '')}</td>
                    <td title="${escapeHtml(record['Infor Mfg Part Num'] || '')}">${escapeHtml(record['Infor Mfg Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Inconsistent Mfg Part Num'] || '')}">${escapeHtml(record['Inconsistent Mfg Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Invalid Buy UOM'] || '')}">${escapeHtml(record['Invalid Buy UOM'] || '')}</td>
                    <td title="${escapeHtml(record['Invalid Item'] || '')}">${escapeHtml(record['Invalid Item'] || '')}</td>
                    <td title="${record['TaskID'] || ''}">${record['TaskID'] || ''}</td>
                `;
                tbody.appendChild(row);
            });
        }
    }

    // Function to update contract to close table
    function updateContractToCloseTable(contractToCloseData) {
        const contractToCloseTable = document.getElementById('contract-to-close-table');
        const contractToCloseNoData = document.getElementById('contract-to-close-no-data');

        if (!contractToCloseTable) {
            console.error('Contract to close table elements not found');
            return;
        }
        
        // Store the data
        paginationState.contractToClose.allData = contractToCloseData;
        paginationState.contractToClose.filteredData = [...contractToCloseData];
        
        // Get table body
        const tbody = contractToCloseTable.querySelector('tbody');
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        // Update count badge - simplified to just show total records
        const contractToCloseCount = document.getElementById('contract-to-close-count');
        if (contractToCloseCount) {
            contractToCloseCount.textContent = `${contractToCloseData.length} records`;
        }
        
        if (contractToCloseData.length === 0) {
            // Show no data message
            if (contractToCloseNoData) contractToCloseNoData.style.display = 'block';
            
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="6" class="no-data-message">No records to display</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message and populate table with all data
            if (contractToCloseNoData) contractToCloseNoData.style.display = 'none';
            
            contractToCloseData.forEach(record => {
                const row = document.createElement('tr');
                
                // Add the row HTML
                row.innerHTML = `
                    <td title="${escapeHtml(record['Contract Number'] || '')}">${escapeHtml(record['Contract Number'] || '')}</td>
                    <td title="${record['Total Lines (Original)'] || '0'}">${record['Total Lines (Original)'] || '0'}</td>
                    <td title="${record['Total Lines (Change Applied)'] || '0'}">${record['Total Lines (Change Applied)'] || '0'}</td>
                    <td title="${record['Total Lines Expired'] || '0'}">${record['Total Lines Expired'] || '0'}</td>
                    <td title="${record['TaskID'] || ''}">${record['TaskID'] || ''}</td>
                    <td title="${escapeHtml(record['UserID'] || '')}">${escapeHtml(record['UserID'] || '')}</td>
                `;
                tbody.appendChild(row);
            });
        }
    }

    // Preview data function
    if (previewBtn) {
        previewBtn.addEventListener('click', function() {
            console.log('Preview button clicked');

            if (loadingSpinner) {
                loadingSpinner.style.display = 'flex';
            }

            // Prepare data for the request
            const requestData = {
                skip_gpo: skipGpoCheckbox.checked,
                export_format: exportFormatSelect.value
            };

            // Fetch data preview
            fetch(getApiUrl('/data-export/preview-data'), {
                method: 'POST',
                body: JSON.stringify(requestData),
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => {
                console.log('Response status:', response.status);
                return response.json();
            })
            .then(data => {
                loadingSpinner.style.display = 'none';

                if (data.success) {

                    // Show the preview section
                    const dataPreview = document.getElementById('data-preview');
                    if (dataPreview) {
                        dataPreview.style.display = 'block';
                    }

                    // Show the horizontal line
                    const horizontalLine = document.getElementById('horizontal-line');
                    if (horizontalLine) {
                        horizontalLine.style.display = 'block';
                    }

                    // show the item and contract to close previews
                    if (itemLinkPreview) {
                        itemLinkPreview.style.display = 'block';
                    }

                    if (contractToClosePreview) {
                        contractToClosePreview.style.display = 'block';
                    }

                    // Show the export button
                    if (exportAllBtn) {
                        exportAllBtn.style.display = 'block';
                    }

                    // Reset pagination state for batch and single tables
                    paginationState.batch.currentPage = 1;
                    paginationState.single.currentPage = 1;

                    // Store all data in pagination state
                    paginationState.batch.allData = data.batch_data || [];
                    paginationState.batch.filteredData = [...paginationState.batch.allData];
                    
                    paginationState.single.allData = data.single_data || [];
                    paginationState.single.filteredData = [...paginationState.single.allData];
                    
                    paginationState.itemLink.allData = data.item_link_data || [];
                    paginationState.itemLink.filteredData = [...paginationState.itemLink.allData];
                    
                    paginationState.contractToClose.allData = data.contract_to_close_data || [];
                    paginationState.contractToClose.filteredData = [...paginationState.contractToClose.allData];

                    // Update tables
                    updateBatchTable(paginationState.batch.allData);
                    updateSingleTable(paginationState.single.allData);
                    updateItemLinkTable(paginationState.itemLink.allData);
                    updateContractToCloseTable(paginationState.contractToClose.allData);

                    // Enable export button if there's data
                    if (exportAllBtn && (
                        paginationState.batch.allData.length > 0 || 
                        paginationState.single.allData.length > 0 ||
                        paginationState.itemLink.allData.length > 0 ||
                        paginationState.contractToClose.allData.length > 0
                    )) {
                        exportAllBtn.disabled = false;
                    } else if (exportAllBtn) {
                        exportAllBtn.disabled = true;
                        // Show the appropriate step complete button if no data
                        const completeStepContainer = document.getElementById('complete-step-container');
                        if (completeStepContainer) {
                            completeStepContainer.style.display = 'block';
                        }
                    }

                    // Scroll to the preview section
                    dataPreview.scrollIntoView({ behavior: 'smooth' });
                } else {
                    alert('Error loading preview data: ' + data.message);
                }
            })
            .catch(error => {
                loadingSpinner.style.display = 'none';
                console.error('Error:', error);
                alert('An error occurred while loading preview data. Please try again.');
            });
        });
    }

    // Update pagination info for batch table
    function updateBatchPagination() {
        const totalItems = paginationState.batch.filteredData.length;
        paginationState.batch.totalPages = Math.max(1, Math.ceil(totalItems / ROWS_PER_PAGE));
        
        // Update pagination controls
        if (paginationControls.batch.totalPagesEl) {
            paginationControls.batch.totalPagesEl.textContent = paginationState.batch.totalPages;
        }
        if (paginationControls.batch.currentPageEl) {
            paginationControls.batch.currentPageEl.textContent = paginationState.batch.currentPage;
        }
        
        // Update button states
        if (paginationControls.batch.prevButton) {
            paginationControls.batch.prevButton.disabled = paginationState.batch.currentPage <= 1;
        }
        if (paginationControls.batch.nextButton) {
            paginationControls.batch.nextButton.disabled = 
                paginationState.batch.currentPage >= paginationState.batch.totalPages;
        }
        
        // Update count badge
        const batchCount = document.getElementById('batch-count');
        if (batchCount) {
            batchCount.textContent = `${totalItems} records`;
        }
    }
    
    // Render current page of batch table
    function renderBatchTablePage() {
        const batchTable = document.getElementById('batch-table');
        const batchNoData = document.getElementById('batch-no-data');
        
        if (!batchTable) return;
        
        const tbody = batchTable.querySelector('tbody');
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        // Calculate start and end indices for current page
        const startIdx = (paginationState.batch.currentPage - 1) * ROWS_PER_PAGE;
        const endIdx = Math.min(startIdx + ROWS_PER_PAGE, paginationState.batch.filteredData.length);
        
        // Get current page data
        const currentPageData = paginationState.batch.filteredData.slice(startIdx, endIdx);
        
        // Update pagination controls
        updateBatchPagination();
        
        if (currentPageData.length === 0) {
            // Show no data message
            if (batchNoData) batchNoData.style.display = 'block';
            
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="21" class="no-data-message">No records to display</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message and populate table
            if (batchNoData) batchNoData.style.display = 'none';
            
            currentPageData.forEach(record => {
                const row = document.createElement('tr');

                // Check for date conflicts
                const hasDateConflict = 
                    (record['Final Date L Check'] && record['Final Date L Check'] !== 'pass') || 
                    (record['Final Date H Check'] && record['Final Date H Check'] !== 'pass');

                // Check for missing vendor part with additional condition
                const missingVendorPart = !record['Vendor Part Num'] && 
                    ['Create', 'Expire then Create (Create)', 'Update (New)'].includes(record['Actual Action']);
                
                // Check for possible duplicates (Create Count > 1)
                const possibleDuplicate = record['Create Count'] && parseInt(record['Create Count']) > 1;

                // Apply appropriate class (prioritize date conflict over possible duplicate over missing vendor part)
                if (hasDateConflict) {
                    row.classList.add('date-conflict');
                } else if (possibleDuplicate) {
                    row.classList.add('possible-duplicate');
                } else if (missingVendorPart) {
                    row.classList.add('missing-vendor-part');
                }

                // Add the row HTML
                row.innerHTML = `
                    <td title="${escapeHtml(record.Organization || '')}">${escapeHtml(record.Organization || '')}</td>
                    <td title="${escapeHtml(record.Vendor || '')}">${escapeHtml(record.Vendor || '')}</td>
                    <td title="${escapeHtml(record.Manufacturer || '')}">${escapeHtml(record.Manufacturer || '')}</td>
                    <td title="${escapeHtml(record['Contract Number'] || '')}">${escapeHtml(record['Contract Number'] || '')}</td>
                    <td title="${escapeHtml(record['Contract Description'] || '')}">${escapeHtml(record['Contract Description'] || '')}</td>
                    <td title="${escapeHtml(record['Tier Level'] || '')}">${escapeHtml(record['Tier Level'] || '')}</td>
                    <td title="${escapeHtml(record['Tier Description'] || '')}">${escapeHtml(record['Tier Description'] || '')}</td>
                    <td title="${escapeHtml(record['Source Type'] || '')}">${escapeHtml(record['Source Type'] || '')}</td>
                    <td title="${formatDate(record['Start Date'])}">${formatDate(record['Start Date'])}</td>
                    <td title="${formatDate(record['End Date'])}">${formatDate(record['End Date'])}</td>
                    <td title="${escapeHtml(record['Mfg Part Num'] || '')}">${escapeHtml(record['Mfg Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Vendor Part Num'] || '')}">${escapeHtml(record['Vendor Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Buyer Part Num'] || '')}">${escapeHtml(record['Buyer Part Num'] || '')}</td>
                    <td class="description-col" title="${escapeHtml(record.Description || '')}">${escapeHtml(record.Description || '')}</td>
                    <td title="${record['Contract Price'] || ''}">${record['Contract Price'] || ''}</td>
                    <td title="${escapeHtml(record.UOM || '')}">${escapeHtml(record.UOM || '')}</td>
                    <td title="${record.QOE || ''}">${record.QOE || ''}</td>
                    <td title="${formatDate(record['Effective Date'])}">${formatDate(record['Effective Date'])}</td>
                    <td title="${formatDate(record['Expiration Date'])}">${formatDate(record['Expiration Date'])}</td>
                    <td class="non-export-col" title="${escapeHtml(record['Actual Action'] || '')}">${escapeHtml(record['Actual Action'] || '')}</td>
                    <td class="non-export-col" title="${escapeHtml(record['ERP Vendor ID (CCX Sync)'] || '')}">${escapeHtml(record['ERP Vendor ID (CCX Sync)'] || '')}</td>
                `;
                tbody.appendChild(row);
            });
        }
    }

    // Function to update batch table with data
    function updateBatchTable(batchData) {
        const batchTable = document.getElementById('batch-table');
        const batchNoData = document.getElementById('batch-no-data');

        if (!batchTable) {
            console.error('Batch table elements not found');
            return;
        }
        
        // Store the data
        paginationState.batch.allData = batchData;
        paginationState.batch.filteredData = [...batchData];
        paginationState.batch.currentPage = 1;
        
        if (batchData.length === 0) {
            // Show no data message
            if (batchNoData) batchNoData.style.display = 'block';

            // Disable and clear batch contract filter
            if (batchContractFilterSelect) {
                batchContractFilterSelect.innerHTML = '<option value="all">All Contracts</option>';
                batchContractFilterSelect.disabled = true;
            }

            const explanationDiv = document.querySelector('.card-body .highlight-explanation');
            if (explanationDiv) {
                explanationDiv.style.display = 'none';
            }
        } else {
            // Populate batch contract filter dropdown
            if (batchContractFilterSelect) {
                // Get unique contract numbers
                const uniqueContracts = [...new Set(batchData.map(record => record['Contract Number']))];

                // Clear previous options except "All Contracts"
                batchContractFilterSelect.innerHTML = '<option value="all">All Contracts</option>';

                // Add each contract as an option
                uniqueContracts.forEach(contract => {
                    if (contract) { // Skip empty values
                        const option = document.createElement('option');
                        option.value = contract;
                        option.textContent = contract;
                        batchContractFilterSelect.appendChild(option);
                    }
                });

                // Enable the dropdown
                batchContractFilterSelect.disabled = false;
            }
        }
        
        // Update pagination and render first page
        updateBatchPagination();
        renderBatchTablePage();
    }

    // function to do filter single table by contract number
    function filterTableByContract(contractNumber) {
        const singleTable = document.getElementById('single-table');
        if (!singleTable) return;
        
        // Reset to page 1 when filtering
        paginationState.single.currentPage = 1;
        
        // If "all" is selected, show all rows
        if (contractNumber === 'all') {
            paginationState.single.filteredData = [...paginationState.single.allData];
        } else {
            // Filter the data based on contract number
            paginationState.single.filteredData = paginationState.single.allData.filter(record => 
                record['Contract Number (PrP)'] === contractNumber
            );
        }
        
        // Update pagination and render
        updateSinglePagination();
        renderSingleTablePage();
    }

    // Update pagination info for single table
    function updateSinglePagination() {
        const totalItems = paginationState.single.filteredData.length;
        paginationState.single.totalPages = Math.max(1, Math.ceil(totalItems / ROWS_PER_PAGE));
        
        // Update pagination controls
        if (paginationControls.single.totalPagesEl) {
            paginationControls.single.totalPagesEl.textContent = paginationState.single.totalPages;
        }
        if (paginationControls.single.currentPageEl) {
            paginationControls.single.currentPageEl.textContent = paginationState.single.currentPage;
        }
        
        // Update button states
        if (paginationControls.single.prevButton) {
            paginationControls.single.prevButton.disabled = paginationState.single.currentPage <= 1;
        }
        if (paginationControls.single.nextButton) {
            paginationControls.single.nextButton.disabled = 
                paginationState.single.currentPage >= paginationState.single.totalPages;
        }
        
        // Update count badge
        const singleCount = document.getElementById('single-count');
        if (singleCount) {
            singleCount.textContent = `${totalItems} records`;
        }
    }
    
    // Render current page of single table
    function renderSingleTablePage() {
        const singleTable = document.getElementById('single-table');
        const singleNoData = document.getElementById('single-no-data');
        
        if (!singleTable) return;
        
        const tbody = singleTable.querySelector('tbody');
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        // Calculate start and end indices for current page
        const startIdx = (paginationState.single.currentPage - 1) * ROWS_PER_PAGE;
        const endIdx = Math.min(startIdx + ROWS_PER_PAGE, paginationState.single.filteredData.length);
        
        // Get current page data
        const currentPageData = paginationState.single.filteredData.slice(startIdx, endIdx);
        
        // Update pagination controls
        updateSinglePagination();
        
        if (currentPageData.length === 0) {
            // Show no data message
            if (singleNoData) singleNoData.style.display = 'block';
            
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="11" class="no-data-message">No records to display</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message and populate table
            if (singleNoData) singleNoData.style.display = 'none';
            
            currentPageData.forEach(record => {
                const row = document.createElement('tr');

                // Check for date conflicts
                const hasDateConflict = 
                    (record['Final Date L Check'] && record['Final Date L Check'] !== 'pass') || 
                    (record['Final Date H Check'] && record['Final Date H Check'] !== 'pass');

                // Check for missing vendor part with additional condition
                const missingVendorPart = !record['Vendor Part Num'] && 
                    ['Create', 'Expire then Create (Create)', 'Update (New)'].includes(record['Actual Action']);
                
                // Check for possible duplicates (Create Count > 1)
                const possibleDuplicate = record['Create Count'] && parseInt(record['Create Count']) > 1;

                // Apply appropriate class (prioritize date conflict over possible duplicate over missing vendor part)
                if (hasDateConflict) {
                    row.classList.add('date-conflict');
                } else if (possibleDuplicate) {
                    row.classList.add('possible-duplicate');
                } else if (missingVendorPart) {
                    row.classList.add('missing-vendor-part');
                }

                row.innerHTML = `
                    <td title="${escapeHtml(record['Mfg Part Num'] || '')}">${escapeHtml(record['Mfg Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Vendor Part Num'] || '')}">${escapeHtml(record['Vendor Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Buyer Part Num'] || '')}">${escapeHtml(record['Buyer Part Num'] || '')}</td>
                    <td class="description-col" title="${escapeHtml(record.Description || '')}">${escapeHtml(record.Description || '')}</td>
                    <td title="${record['Contract Price'] || ''}">${record['Contract Price'] || ''}</td>
                    <td title="${escapeHtml(record.UOM || '')}">${escapeHtml(record.UOM || '')}</td>
                    <td title="${record.QOE || ''}">${record.QOE || ''}</td>
                    <td title="${formatDate(record['Effective Date'])}">${formatDate(record['Effective Date'])}</td>
                    <td title="${formatDate(record['Expiration Date'])}">${formatDate(record['Expiration Date'])}</td>
                    <td class="non-export-col" title="${escapeHtml(record['Contract Number (PrP)'] || '')}">${escapeHtml(record['Contract Number (PrP)'] || '')}</td>
                    <td class="non-export-col" title="${escapeHtml(record['ERP Vendor ID (PrP)'] || '')}">${escapeHtml(record['ERP Vendor ID (PrP)'] || '')}</td>
                `;
                tbody.appendChild(row);
            });
        }
    }

    // Function to update single table with data
    function updateSingleTable(singleData) {
        const singleTable = document.getElementById('single-table');
        const singleNoData = document.getElementById('single-no-data');
        const contractFilterSelect = document.getElementById('contract-filter');

        if (!singleTable) {
            console.error('Single table elements not found');
            return;
        }
        
        // Store the data
        paginationState.single.allData = singleData;
        paginationState.single.filteredData = [...singleData];
        paginationState.single.currentPage = 1;
        
        if (singleData.length === 0) {
            // Show no data message
            if (singleNoData) singleNoData.style.display = 'block';

            // Disable and clear contract filter
            if (contractFilterSelect) {
                contractFilterSelect.innerHTML = '<option value="all">All Contracts</option>';
                contractFilterSelect.disabled = true;
            }

            const explanationDiv = document.querySelector('.card-body .highlight-explanation');
            if (explanationDiv) {
                explanationDiv.style.display = 'none';
            }
        } else {
            // Populate contract filter dropdown
            if (contractFilterSelect) {
                // Get unique contract numbers
                const uniqueContracts = [...new Set(singleData.map(record => record['Contract Number (PrP)']))];

                // Clear previous options except "All Contracts"
                contractFilterSelect.innerHTML = '<option value="all">All Contracts</option>';

                // Add each contract as an option
                uniqueContracts.forEach(contract => {
                    if (contract) { // Skip empty values
                        const option = document.createElement('option');
                        option.value = contract;
                        option.textContent = contract;
                        contractFilterSelect.appendChild(option);
                    }
                });

                // Enable the dropdown
                contractFilterSelect.disabled = false;
            }
        }
        
        // Update pagination and render first page
        updateSinglePagination();
        renderSingleTablePage();
    }
    
    

    // Function to update contract to close table
    function updateContractToCloseTable(contractToCloseData) {
        const contractToCloseTable = document.getElementById('contract-to-close-table');
        const contractToCloseNoData = document.getElementById('contract-to-close-no-data');

        if (!contractToCloseTable) {
            console.error('Contract to close table elements not found');
            return;
        }
        
        // Store the datawht d
        paginationState.contractToClose.allData = contractToCloseData;
        paginationState.contractToClose.filteredData = [...contractToCloseData];
        
        // Get table body
        const tbody = contractToCloseTable.querySelector('tbody');
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        // Update count badge - simplified to just show total records
        const contractToCloseCount = document.getElementById('contract-to-close-count');
        if (contractToCloseCount) {
            contractToCloseCount.textContent = `${contractToCloseData.length} records`;
        }
        
        if (contractToCloseData.length === 0) {
            // Show no data message
            if (contractToCloseNoData) contractToCloseNoData.style.display = 'block';
            
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="6" class="no-data-message">No records to display</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message and populate table with all data
            if (contractToCloseNoData) contractToCloseNoData.style.display = 'none';
            
            contractToCloseData.forEach(record => {
                const row = document.createElement('tr');
                
                // Add the row HTML
                row.innerHTML = `
                    <td title="${escapeHtml(record['Contract Number'] || '')}">${escapeHtml(record['Contract Number'] || '')}</td>
                    <td title="${record['Total Lines (Original)'] || '0'}">${record['Total Lines (Original)'] || '0'}</td>
                    <td title="${record['Total Lines (Change Applied)'] || '0'}">${record['Total Lines (Change Applied)'] || '0'}</td>
                    <td title="${record['Total Lines Expired'] || '0'}">${record['Total Lines Expired'] || '0'}</td>
                    <td title="${record['TaskID'] || ''}">${record['TaskID'] || ''}</td>
                    <td title="${escapeHtml(record['UserID'] || '')}">${escapeHtml(record['UserID'] || '')}</td>
                `;
                tbody.appendChild(row);
            });
        }
    }

    // Helper function to escape HTML
    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Helper function to format dates
    function formatDate(dateStr) {
        if (!dateStr) return '';
        try {
            // Handle different date formats
            const date = new Date(dateStr);
            if (isNaN(date.getTime())) return dateStr;
            return date.toLocaleDateString();
        } catch (e) {
            return dateStr;
        }
    }
});