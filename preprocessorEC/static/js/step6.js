document.addEventListener('DOMContentLoaded', function() {
    
    // Cache DOM elements
    const previewBtn = document.getElementById('preview-btn');
    const dataPreview = document.getElementById('data-preview');
    const exportResults = document.getElementById('export-results');
    const loadingSpinner = document.getElementById('loading-spinner');
    const exportBatchBtn = document.getElementById('export-batch-btn');
    const exportSingleBtn = document.getElementById('export-single-btn');
    const skipGpoCheckbox = document.getElementById('skip_gpo');
    const exportFormatSelect = document.getElementById('export_format');
    
    // Store data after preview
    let exportData = null;
    
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
                    // Store the data for export later
                    exportData = data;
                    
                    // Show the preview section
                    dataPreview.style.display = 'block';
                    
                    // Update batch table
                    updateBatchTable(data.batch_data || []);
                    
                    // Update single contract table
                    updateSingleTable(data.single_data || []);
                    
                    // Enable export buttons if there's data and user has permission
                    if (exportBatchBtn && data.batch_data && data.batch_data.length > 0) {
                        exportBatchBtn.disabled = false;
                    }
                    
                    if (exportSingleBtn && data.single_data && data.single_data.length > 0) {
                        exportSingleBtn.disabled = false;
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
    
    // Export batch data function is fine, no changes needed
    // Export single contract data function is fine, no changes needed
    
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
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="16" class="no-data-message">No batch upload records available</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message and populate table
            if (batchNoData) batchNoData.style.display = 'none';
            
            batchData.forEach(record => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td title="${escapeHtml(record.Organization || '')}">${escapeHtml(record.Organization || '')}</td>
                    <td title="${escapeHtml(record.Vendor || '')}">${escapeHtml(record.Vendor || '')}</td>
                    <td title="${escapeHtml(record.Manufacturer || '')}">${escapeHtml(record.Manufacturer || '')}</td>
                    <td title="${escapeHtml(record['Contract Number'] || '')}">${escapeHtml(record['Contract Number'] || '')}</td>
                    <td title="${escapeHtml(record['Contract Description'] || '')}">${escapeHtml(record['Contract Description'] || '')}</td>
                    <td title="${escapeHtml(record['Tier Level'] || '')}">${escapeHtml(record['Tier Level'] || '')}</td>
                    <td title="${escapeHtml(record['Source Type'] || '')}">${escapeHtml(record['Source Type'] || '')}</td>
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
        }
    }
    
    // Function to update single contract table with data
    function updateSingleTable(singleData) {
        const singleTable = document.getElementById('single-table');
        const singleCount = document.getElementById('single-count');
        const singleNoData = document.getElementById('single-no-data');
        
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
            const noDataRow = document.createElement('tr');
            noDataRow.className = 'no-data-row';
            noDataRow.innerHTML = '<td colspan="10" class="no-data-message">No single contract records available</td>';
            tbody.appendChild(noDataRow);
        } else {
            // Hide no data message and populate table
            if (singleNoData) singleNoData.style.display = 'none';
            
            singleData.forEach(record => {
                const row = document.createElement('tr');
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
        }
    }
    
    // Function to perform the actual export - no changes needed
    
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