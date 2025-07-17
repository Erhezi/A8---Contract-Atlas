document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const previewBtn = document.getElementById('preview-btn');
    const exportAllBtn = document.getElementById('export-all-btn');
    const loadingSpinner = document.getElementById('loading-spinner');
    const skipGpoCheckbox = document.getElementById('skip_gpo');
    const exportFormatSelect = document.getElementById('export_format');
    const batchContractFilterSelect = document.getElementById('batch-contract-filter');
    const contractFilterSelect = document.getElementById('contract-filter');
    const exportResults = document.getElementById('export-results');
    const fileLinksContainer = document.getElementById('file-links-container');
    const itemLinkPreview = document.getElementById('item-link-preview');
    const contractToClosePreview = document.getElementById('contract-to-close-preview');

    // If batch contract filter exists, add event listener
    if (batchContractFilterSelect) {
        batchContractFilterSelect.addEventListener('change', function() {
            const selectedContract = this.value;
            filterBatchTableByContract(selectedContract);
        });
    }

    console.log('batchContractFilterSelect:', batchContractFilterSelect);

    // If contract filter exists, add event listener
    if (contractFilterSelect) {
        contractFilterSelect.addEventListener('change', function() {
            const selectedContract = this.value;
            filterTableByContract(selectedContract);
        });
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

                    // Update batch table
                    updateBatchTable(data.batch_data || []);

                    // Update single contract table
                    updateSingleTable(data.single_data || []);
                    
                    // Update item link table (new)
                    updateItemLinkTable(data.item_link_data || []);
                    
                    // Update contract to close table (new)
                    updateContractToCloseTable(data.contract_to_close_data || []);

                    // Enable export button if there's data
                    if (exportAllBtn && (
                        (data.batch_data && data.batch_data.length > 0) || 
                        (data.single_data && data.single_data.length > 0) ||
                        (data.item_link_data && data.item_link_data.length > 0) ||
                        (data.contract_to_close_data && data.contract_to_close_data.length > 0)
                    )) {
                        exportAllBtn.disabled = false;
                    } else if (exportAllBtn) {
                        exportAllBtn.disabled = true;
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

    // Export all data function
    if (exportAllBtn) {
        exportAllBtn.addEventListener('click', function() {
            if (loadingSpinner) {
                loadingSpinner.style.display = 'flex';
            }

            // Prepare data for the request
            const requestData = {
                skip_gpo: skipGpoCheckbox.checked,
                export_format: exportFormatSelect.value,
                export_type: 'all' // Export all types at once
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
                loadingSpinner.style.display = 'none';

                if (data.success) {
                    // Show export results section
                    exportResults.style.display = 'block';

                    // Clear previous file links
                    fileLinksContainer.innerHTML = '';

                    // Add link to the ZIP file
                    if (data.zipFile) {
                        const zipLinkHtml = `
                            <div class="file-link">
                                <a href="${data.zipFile.url}" download class="btn btn-sm btn-outline-primary">
                                    <i class="fas fa-file-archive"></i> ${data.zipFile.name}
                                </a>
                                <span class="ml-2 text-muted">${data.zipFile.description || 'All export files'}</span>
                            </div>
                        `;
                        fileLinksContainer.innerHTML += zipLinkHtml;
                    }

                    // Scroll to the results section
                    exportResults.scrollIntoView({ behavior: 'smooth' });
                } else {
                    alert('Export failed: ' + data.message);
                }
            })
            .catch(error => {
                loadingSpinner.style.display = 'none';
                console.error('Error:', error);
                alert('An error occurred during export. Please try again.');
            });
        });
    }

    // function to do filter batch table by contract number
    function filterBatchTableByContract(contractNumber) {
        const batchTable = document.getElementById('batch-table');
        if (!batchTable) return;
        
        const rows = batchTable.querySelectorAll('tbody tr');
        
        // If "all" is selected, show all rows
        if (contractNumber === 'all') {
            rows.forEach(row => {
                row.style.display = '';
            });
            return;
        }
        
        // Otherwise, filter rows based on the selected contract
        rows.forEach(row => {
            // Contract number is in the fourth column
            const contractCell = row.querySelector('td:nth-child(4)');
            if (contractCell) {
                const rowContractNumber = contractCell.textContent || '';
                if (rowContractNumber === contractNumber) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            }
        });
        
        // Update the count badge to show filtered count
        const visibleRows = Array.from(rows).filter(row => row.style.display !== 'none');
        const batchCount = document.getElementById('batch-count');
        if (batchCount) {
            batchCount.textContent = `${visibleRows.length} records`;
        }
    }

    // Function to update batch table with data
    function updateBatchTable(batchData) {
        const batchTable = document.getElementById('batch-table');
        const batchCount = document.getElementById('batch-count');
        const batchNoData = document.getElementById('batch-no-data');

        if (!batchTable || !batchCount) {
            console.error('Batch table elements not found');
            return;
        }

        const tbody = batchTable.querySelector('tbody');

        // Clear existing rows
        tbody.innerHTML = '';

        // Update count badge
        batchCount.textContent = `${batchData.length} records`;

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
            // Hide no data message and populate table
            if (batchNoData) batchNoData.style.display = 'none';

            batchData.forEach(record => {
                const row = document.createElement('tr');

                // Check for date conflicts
                const hasDateConflict = 
                    (record['Final Date L Check'] && record['Final Date L Check'] !== 'pass') || 
                    (record['Final Date H Check'] && record['Final Date H Check'] !== 'pass');

                // Check for missing vendor part with additional condition
                const missingVendorPart = !record['Vendor Part Num'] && 
                    ['Create', 'Expire then Create (Create)', 'Update (New)'].includes(record['Actual Action']);

                // Apply appropriate class (prioritize date conflict over missing vendor part)
                if (hasDateConflict) {
                    row.classList.add('date-conflict');
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
                `;
                tbody.appendChild(row);
            });

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
    }

    // filter by contract number for better user experience
    function filterTableByContract(contractNumber) {
        const singleTable = document.getElementById('single-table');
        if (!singleTable) return;
        
        const rows = singleTable.querySelectorAll('tbody tr');
        
        // If "all" is selected, show all rows
        if (contractNumber === 'all') {
            rows.forEach(row => {
                row.style.display = '';
            });
            return;
        }
        
        // Otherwise, filter rows based on the selected contract
        rows.forEach(row => {
            // First column contains the contract number
            const contractCell = row.querySelector('td:first-child');
            if (contractCell) {
                const rowContractNumber = contractCell.textContent || '';
                if (rowContractNumber === contractNumber) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            }
        });
        
        // Update the count badge to show filtered count
        const visibleRows = Array.from(rows).filter(row => row.style.display !== 'none');
        const singleCount = document.getElementById('single-count');
        if (singleCount) {
            singleCount.textContent = `${visibleRows.length} records`;
        }
    }

    // Function to update single contract table with data
    function updateSingleTable(singleData) {
        const singleTable = document.getElementById('single-table');
        const singleCount = document.getElementById('single-count');
        const singleNoData = document.getElementById('single-no-data');
        const contractFilterSelect = document.getElementById('contract-filter');

        if (!singleTable || !singleCount) {
            console.error('Single table elements not found');
            return;
        }

        const tbody = singleTable.querySelector('tbody');

        // Clear existing rows
        tbody.innerHTML = '';

        // Update count badge
        singleCount.textContent = `${singleData.length} records`;

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
            // Hide no data message
            if (singleNoData) singleNoData.style.display = 'none';

            // Populate table with data
            singleData.forEach(record => {
                const row = document.createElement('tr');

                // Check for date conflicts (if present in single contract data)
                const hasDateConflict = 
                    (record['Final Date L Check'] && record['Final Date L Check'] !== 'pass') || 
                    (record['Final Date H Check'] && record['Final Date H Check'] !== 'pass');

                // Check for missing vendor part with additional condition
                const missingVendorPart = !record['Vendor Part Num'] && 
                    ['Create', 'Expire then Create (Create)', 'Update (New)'].includes(record['Actual Action']);

                // Apply appropriate class
                if (hasDateConflict) {
                    row.classList.add('date-conflict');
                } else if (missingVendorPart) {
                    row.classList.add('missing-vendor-part');
                }

                row.innerHTML = `
                    <td title="${escapeHtml(record['Contract Number (PrP)'] || '')}">${escapeHtml(record['Contract Number (PrP)'] || '')}</td>
                    <td title="${escapeHtml(record['Mfg Part Num'] || '')}">${escapeHtml(record['Mfg Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Vendor Part Num'] || '')}">${escapeHtml(record['Vendor Part Num'] || '')}</td>
                    <td title="${escapeHtml(record['Buyer Part Num'] || '')}">${escapeHtml(record['Buyer Part Num'] || '')}</td>
                    <td class="description-col" title="${escapeHtml(record.Description || '')}">${escapeHtml(record.Description || '')}</td>
                    <td title="${record['Contract Price'] || ''}">${record['Contract Price'] || ''}</td>
                    <td title="${escapeHtml(record.UOM || '')}">${escapeHtml(record.UOM || '')}</td>
                    <td title="${record.QOE || ''}">${record.QOE || ''}</td>
                    <td title="${formatDate(record['Effective Date'])}">${formatDate(record['Effective Date'])}</td>
                    <td title="${formatDate(record['Expiration Date'])}">${formatDate(record['Expiration Date'])}</td>
                `;
                tbody.appendChild(row);
            });

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
    }

    // Function to update item link table
    function updateItemLinkTable(itemLinkData) {
        const itemLinkTable = document.getElementById('item-link-table');
        const itemLinkCount = document.getElementById('item-link-count');
        const itemLinkNoData = document.getElementById('item-link-no-data');

        if (!itemLinkTable || !itemLinkCount) {
            console.error('Item link table elements not found');
            return;
        }

        const tbody = itemLinkTable.querySelector('tbody');

        // Clear existing rows
        tbody.innerHTML = '';

        // Update count badge
        itemLinkCount.textContent = `${itemLinkData.length} records`;

        if (itemLinkData.length === 0) {
            // Show no data message
            if (itemLinkNoData) itemLinkNoData.style.display = 'block';
            
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="14" class="no-data-message">No item link records available</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message
            if (itemLinkNoData) itemLinkNoData.style.display = 'none';

            // Populate table with data
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
        const contractToCloseCount = document.getElementById('contract-to-close-count');
        const contractToCloseNoData = document.getElementById('contract-to-close-no-data');

        if (!contractToCloseTable || !contractToCloseCount) {
            console.error('Contract to close table elements not found');
            return;
        }

        const tbody = contractToCloseTable.querySelector('tbody');

        // Clear existing rows
        tbody.innerHTML = '';

        // Update count badge
        contractToCloseCount.textContent = `${contractToCloseData.length} records`;

        if (contractToCloseData.length === 0) {
            // Show no data message
            if (contractToCloseNoData) contractToCloseNoData.style.display = 'block';
            
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="6" class="no-data-message">No contracts to close available</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message
            if (contractToCloseNoData) contractToCloseNoData.style.display = 'none';

            // Populate table with data
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