document.addEventListener('DOMContentLoaded', function() {
    // Update file input label with selected filename
    document.getElementById('file').addEventListener('change', function(e) {
        const fileName = e.target.files[0] ? e.target.files[0].name : 'Choose file';
        const fileLabel = document.querySelector('.custom-file-label');
        fileLabel.textContent = fileName;
        
        // Change upload button color when file is selected
        const uploadBtn = document.getElementById('file-upload-btn');
        if (e.target.files[0]) {
            uploadBtn.classList.add('file-selected');
        } else {
            uploadBtn.classList.remove('file-selected');
        }
    });
    
    // Make the Browse button functional
    document.getElementById('file-browsing-btn').addEventListener('click', function() {
        document.getElementById('file').click();
    });

    // File Upload Form
    document.getElementById('upload-form').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const uploadBtn = this.querySelector('button[type="submit"]');
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Uploading...';
        
        fetch(getApiUrl('/file-processing/upload-file'), {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload File';
            
            if (data.success) {
                // Show mapping container
                document.getElementById('upload-container').innerHTML = `
                    <div class="alert alert-success">
                        <strong>File uploaded:</strong> ${data.filename}
                    </div>
                `;
                
                // Define column name variations for auto-mapping
                const columnMappings = {
                    'Mfg Part Num': ['mfg part num', 'mfg part number', 'manufacturer part', 'manufacturer part number', 'mfr part', 'mfrpartnum', 'mfg part #', 'manufacturer part #', 'manufacturer number'],
                    'Vendor Part Num': ['vendor part', 'vendor part num', 'vendor part number', 'supplier part', 'supplierpartnum', 'vendor part #', 'supplier part #', 'vendor item'],
                    'Buyer Part Num': ['buyer part', 'buyer part num', 'buyer part number', 'customer part', 'customerpartnum', 'buyer part #', 'customer part #'],
                    'Description': ['description', 'desc', 'item desc', 'product desc', 'item description'],
                    'Contract Price': ['contract price', 'price', 'unit price', 'cost', 'contract cost'],
                    'UOM': ['uom', 'unit of measure', 'unit', 'measure', 'unit measure'],
                    'QOE': ['qoe', 'quantity of each', 'qty each', 'package qty', 'qty per package'],
                    'Effective Date': ['effective date', 'start date', 'valid from', 'start', 'begin date'],
                    'Expiration Date': ['expiration date', 'exp date', 'end date', 'valid until', 'termination date'],
                    'Contract Number': ['contract number', 'contract num', 'contract #', 'contract id', 'agreement number'],
                    'ERP Vendor ID': ['erp vendor id', 'vendorid', 'vendor erp id', 'erp id', 'vendor id', 'supplier id', 'erp supplier', 'vendor code'],
                    'Source Contract Type': ['contract source type', 'source contract type', 'contract source', 'source type', 'contract type', 'agreement type', 'agreement source', 'gpo type', 'source', 'gpo or local'],
                    'Intended Action': ['intended action', 'action', 'intended use', 'action type', 'intention', 'upsert expire', 'upsert/expire', 'upsert or expire']
                };
                
                // Normalize headers to lowercase for matching
                const normalizedHeaders = data.headers.map(h => h.toLowerCase());
                
                // Auto-map columns based on similar names
                const autoMappings = {};
                let unmatchedFields = 0;
                
                // Populate dropdowns with file headers
                const selects = document.querySelectorAll('#mapping-form .column-mapping-select');
                selects.forEach(select => {
                    // Clear any existing options
                    select.innerHTML = '';
                    
                    // Add empty option
                    const emptyOption = document.createElement('option');
                    emptyOption.value = '';
                    emptyOption.textContent = '-- Select Column --';
                    select.appendChild(emptyOption);
                    
                    // Add headers as options
                    data.headers.forEach(header => {
                        const option = document.createElement('option');
                        option.value = header;
                        option.textContent = header;
                        select.appendChild(option);
                    });
                    
                    // Try to auto-match this field
                    const fieldName = select.name;
                    if (columnMappings[fieldName]) {
                        // Look for a match in the uploaded headers
                        let bestMatch = null;
                        
                        // First try exact match
                        const exactMatch = normalizedHeaders.findIndex(h => 
                            columnMappings[fieldName].includes(h));
                        
                        if (exactMatch !== -1) {
                            bestMatch = data.headers[exactMatch];
                        } else {
                            // Try partial match with exclusions for ERP Vendor ID
                            for (let i = 0; i < normalizedHeaders.length; i++) {
                                // Skip 'vendor' and 'vendor name' for ERP Vendor ID field
                                if (fieldName === 'ERP Vendor ID' && 
                                    (normalizedHeaders[i] === 'vendor' || normalizedHeaders[i] === 'vendor name')) {
                                    continue;
                                }
                                
                                for (let j = 0; j < columnMappings[fieldName].length; j++) {
                                    if (normalizedHeaders[i].includes(columnMappings[fieldName][j]) || 
                                        columnMappings[fieldName][j].includes(normalizedHeaders[i])) {
                                        bestMatch = data.headers[i];
                                        break;
                                    }
                                }
                                if (bestMatch) break;
                            }
                        }
                        
                        if (bestMatch) {
                            select.value = bestMatch;
                            autoMappings[fieldName] = bestMatch;
                            
                            // Add visual indicator for auto-matched fields
                            const parentDiv = select.closest('.form-group');
                            parentDiv.classList.add('auto-matched');
                            const label = parentDiv.querySelector('label');
                            label.innerHTML += ' <span class="auto-match-indicator">(Auto-matched)</span>';
                        } else if (select.classList.contains('required-field')) {
                            unmatchedFields++;
                        }
                    }
                });
                
                // Show mapping container with a message about auto-matching
                document.getElementById('mapping-container').style.display = 'block';

                // scroll smoothly to the mapping container
                document.getElementById('mapping-container').scrollIntoView({ behavior: 'smooth' });
                
                // Add notification about auto-matching
                const mappingForm = document.getElementById('mapping-form');
                const autoMapNotice = document.createElement('div');
                autoMapNotice.className = 'alert alert-info';
                
                if (Object.keys(autoMappings).length > 0) {
                    autoMapNotice.innerHTML = `
                        <strong>Auto-matched ${Object.keys(autoMappings).length} columns.</strong> 
                        ${unmatchedFields > 0 ? 'Please review and complete the ' + unmatchedFields + ' remaining required fields.' : 'All required fields were matched! Please review and confirm.'}
                    `;
                    mappingForm.insertBefore(autoMapNotice, mappingForm.firstChild);
                }
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            uploadBtn.disabled = false;
            uploadBtn.textContent = 'Upload File';
            alert('Error uploading file: ' + error);
        });
    });


    
    // Column Mapping Form
    document.getElementById('mapping-form').addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Validate that all required fields are mapped
        const requiredSelects = document.querySelectorAll('.required-field');
        let isValid = true;
        
        requiredSelects.forEach(select => {
            if (!select.value) {
                isValid = false;
                select.classList.add('is-invalid');
            } else {
                select.classList.remove('is-invalid');
            }
        });
        
        if (!isValid) {
            alert('Please map all required fields');
            return;
        }
        
        const formData = new FormData(this);
        const mappingBtn = this.querySelector('button[type="submit"]');
        mappingBtn.disabled = true;
        mappingBtn.textContent = 'Processing...';
        
        // First save the column mapping
        fetch(getApiUrl('/file-processing/map-columns'), {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Now validate the file
                return fetch(getApiUrl('/file-processing/validate-file'), {
                    method: 'POST',
                    body: formData
                });
            } else {
                throw new Error(data.message);
            }
        })
        .then(response => response.json())
        .then(data => {
            mappingBtn.disabled = false;
            mappingBtn.textContent = 'Validate';
            
            document.getElementById('validation-container').style.display = 'block';

            // Update stats regardless of success/failure
            if (data.stats) {
                document.getElementById('validation-stats').style.display = 'block';
                
                // Update total rows
                document.getElementById('total-rows').textContent = data.stats.total_rows || 0;
                
                // Update error rows and change color based on value
                const errorRows = data.stats.error_rows || 0;
                document.getElementById('error-rows').textContent = errorRows;
                
                // Error card - change to green if no errors
                const errorCard = document.getElementById('error-card');
                const errorLabel = document.getElementById('error-label');
                const errorCount = document.getElementById('error-rows');
                
                if (errorRows === 0) {
                    // Change to success green
                    errorCard.style.backgroundColor = '#d4edda';
                    errorLabel.style.color = '#155724';
                    errorCount.style.color = '#155724';
                } else {
                    // Keep as error red
                    errorCard.style.backgroundColor = '#f8d7da';
                    errorLabel.style.color = '#721c24';
                    errorCount.style.color = '#721c24';
                }
                
                // Update duplicate rows and change color based on value
                const duplicateRows = data.stats.duplicate_rows || 0;
                document.getElementById('duplicate-rows').textContent = duplicateRows;
                
                // Duplicate card - change to green if no duplicates
                const duplicateCard = document.getElementById('duplicate-card');
                const duplicateLabel = document.getElementById('duplicate-label');
                const duplicateCount = document.getElementById('duplicate-rows');
                
                if (duplicateRows === 0) {
                    // Change to success green
                    duplicateCard.style.backgroundColor = '#d4edda';
                    duplicateLabel.style.color = '#155724';
                    duplicateCount.style.color = '#155724';
                } else {
                    // Keep as warning yellow
                    duplicateCard.style.backgroundColor = '#fff3cd';
                    duplicateLabel.style.color = '#856404';
                    duplicateCount.style.color = '#856404';
                }
            }
            
            if (data.success) {
                // check for missing vendor part numbers in distributor mode
                const duplicateMode = document.getElementById('duplicate_mode').value;
                const missingVendorPartNum = data.stats.missing_vendor_part_num || 0;

                // Build the validation results content
                let successContent = `<div class="alert alert-success">${data.message}</div>`;

                 // Add warning about missing vendor part numbers if in distributor mode
                if (duplicateMode === 'distributor' && missingVendorPartNum > 0) {
                    successContent = `
                        <div class="alert alert-warning">
                            <strong>!!!</strong> ${missingVendorPartNum} ${missingVendorPartNum === 1 ? 'line' : 'lines'} 
                            without Vendor Part Number detected. These items will not be synchronized to Infor.
                        </div>
                        ${successContent}
                    `;
                }

                // Show success message with potential warning
                document.getElementById('validation-results').innerHTML = successContent;
                document.getElementById('error-display').style.display = 'none';
                document.getElementById('success-display').style.display = 'block';
            } else {
                // Show error message and table
                document.getElementById('validation-results').innerHTML = `
                    <div class="alert alert-danger">${data.message}</div>
                `;
                document.getElementById('error-table-container').innerHTML = data.error_table;
                const table = document.querySelector('#error-table-container table');
                if (table) {
                    // Ensure table has appropriate styling
                    table.style.width = 'auto';
                    table.style.minWidth = '100%';
                    table.style.fontSize = '0.8rem'; 
                    
                    // Find and style the Description column
                    const headers = table.querySelectorAll('th');
                    let descriptionIndex = -1;
                    
                    // Find which column is the Description column
                    headers.forEach((header, index) => {
                        if (header.textContent.includes('Description') || header.textContent.includes('Contract Number')) {
                        header.style.maxWidth = '300px';
                        header.style.whiteSpace = 'nowrap';
                        header.style.overflow = 'hidden';
                        header.style.textOverflow = 'ellipsis';
                        
                        // Keep track of Description index specifically
                        if (header.textContent.includes('Description')) {
                            descriptionIndex = index;
                        }
                        
                        // Add handling for Contract Number column
                        if (header.textContent.includes('Contract Number')) {
                            const contractCells = table.querySelectorAll(`td:nth-child(${index + 1})`);
                            contractCells.forEach(cell => {
                                cell.style.maxWidth = '300px';
                                cell.style.whiteSpace = 'nowrap';
                                cell.style.overflow = 'hidden';
                                cell.style.textOverflow = 'ellipsis';
                            });
                        }
                    }
                    // Make all headers consistent
                    header.style.fontSize = '0.8rem';
                    header.style.whiteSpace = 'nowrap';
                });
                    
                    // Style all cells in the Description column
                    if (descriptionIndex !== -1) {
                        const descriptionCells = table.querySelectorAll(`td:nth-child(${descriptionIndex + 1})`);
                        descriptionCells.forEach(cell => {
                            cell.style.maxWidth = '200px';
                            cell.style.whiteSpace = 'nowrap';
                            cell.style.overflow = 'hidden';
                            cell.style.textOverflow = 'ellipsis';
                        });
                    }
                    
                    // Ensure horizontal scrolling works properly
                    const innerContainer = document.getElementById('error-table-container');
                    innerContainer.style.overflowX = 'auto';
                    innerContainer.style.overflowY = 'auto';
                    innerContainer.style.maxHeight = '500px';
                }
                document.getElementById('error-display').style.display = 'block';
                document.getElementById('success-display').style.display = 'none';
                
                // Add styles to highlight error rows
                const errorTable = document.querySelector('#error-table-container table');
                if (errorTable) {
                    const rows = errorTable.querySelectorAll('tbody tr');
                    rows.forEach(row => {
                        row.style.backgroundColor = '#fee';
                    });
                }
            }

            // Scroll to validation container
            document.getElementById('validation-container').scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            mappingBtn.disabled = false;
            mappingBtn.textContent = 'Validate';
            alert('Error: ' + error.message);
        });
    });
});