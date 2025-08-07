/**
 * Step 7: Synchronization Inspection JavaScript
 */

// Global variables
let currentTasks = [];
let selectedTaskId = null;
let hasErrors = false;
let stepCompleted = false;

// DOM elements
const loadingSpinner = document.getElementById('loading-spinner');
const taskHeaderTbody = document.getElementById('task-header-tbody');
const taskFilter = document.getElementById('task-filter');
const taskCountInfo = document.getElementById('task-count-info');
const viewSyncDetailsBtn = document.getElementById('view-sync-details-btn');
const syncDetailsSection = document.getElementById('sync-details-section');
const selectedTaskIdSpan = document.getElementById('selected-task-id');
const completionSection = document.getElementById('completion-section');
const markCcxBtn = document.getElementById('mark-ccx-btn');
const markInforBtn = document.getElementById('mark-infor-btn');
const goHomeBtn = document.getElementById('go-home-btn');
const commentSection = document.getElementById('comment-section');
const completionComment = document.getElementById('completion-comment');
const nextStepBtn = document.getElementById('next-step-btn');

// Custom dropdown elements (will be created dynamically)
let customDropdown = null;
let dropdownToggle = null;
let dropdownMenu = null;
let searchInput = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    createCustomDropdown();
    loadTaskHeaders();
    setupEventListeners();
});

function createCustomDropdown() {
    const wrapper = document.querySelector('.custom-select-wrapper');
    
    // Create custom dropdown structure
    customDropdown = document.createElement('div');
    customDropdown.className = 'custom-dropdown';
    
    dropdownToggle = document.createElement('button');
    dropdownToggle.className = 'custom-dropdown-toggle';
    dropdownToggle.textContent = 'All Tasks - Select Task ID to Inspect';
    dropdownToggle.type = 'button';
    
    dropdownMenu = document.createElement('div');
    dropdownMenu.className = 'custom-dropdown-menu';
    
    searchInput = document.createElement('input');
    searchInput.className = 'dropdown-search-input';
    searchInput.type = 'text';
    searchInput.placeholder = 'Search by Task ID, User ID, or TP Filename...';
    
    dropdownMenu.appendChild(searchInput);
    customDropdown.appendChild(dropdownToggle);
    customDropdown.appendChild(dropdownMenu);
    wrapper.appendChild(customDropdown);
}

function setupEventListeners() {
    // Dropdown toggle
    dropdownToggle.addEventListener('click', function(e) {
        e.stopPropagation();
        toggleDropdown();
    });

    // Search input
    searchInput.addEventListener('input', function() {
        filterDropdownOptions();
    });

    // Handle keyboard navigation in search input
    searchInput.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeDropdown();
        } else if (e.key === 'Enter') {
            e.preventDefault();
            // Select first visible option if any
            const firstVisibleOption = dropdownMenu.querySelector('.dropdown-option:not(.hidden)');
            if (firstVisibleOption && firstVisibleOption.dataset.taskId) {
                firstVisibleOption.click();
            }
        }
    });

    // Prevent dropdown from closing when clicking inside
    dropdownMenu.addEventListener('click', function(e) {
        e.stopPropagation();
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', function() {
        closeDropdown();
    });

    // View sync details button
    viewSyncDetailsBtn.addEventListener('click', function() {
        if (selectedTaskId) {
            loadSyncDetails(selectedTaskId);
        }
    });

    // Completion buttons
    markCcxBtn.addEventListener('click', function() {
        markStepCompleted('CCX');
    });

    markInforBtn.addEventListener('click', function() {
        markStepCompleted('Infor');
    });

    // Go home button
    goHomeBtn.addEventListener('click', function() {
        window.location.href = '/common/home';
    });

    // Next step button
    nextStepBtn.addEventListener('click', function() {
        proceedToNextStep();
    });
}

function toggleDropdown() {
    const isOpen = dropdownMenu.classList.contains('show');
    
    if (isOpen) {
        closeDropdown();
    } else {
        openDropdown();
    }
}

function openDropdown() {
    dropdownMenu.classList.add('show');
    dropdownToggle.classList.add('open');
    searchInput.focus();
}

function closeDropdown() {
    dropdownMenu.classList.remove('show');
    dropdownToggle.classList.remove('open');
    searchInput.value = '';
    filterDropdownOptions(); // Reset filter
}

function filterDropdownOptions() {
    const searchTerm = searchInput.value.toLowerCase();
    const options = dropdownMenu.querySelectorAll('.dropdown-option');
    let visibleCount = 0;
    
    options.forEach(option => {
        const text = option.textContent.toLowerCase();
        const shouldShow = text.includes(searchTerm);
        
        if (shouldShow) {
            option.classList.remove('hidden');
            visibleCount++;
        } else {
            option.classList.add('hidden');
        }
    });
    
    // Show/hide no results message
    let noResultsMsg = dropdownMenu.querySelector('.no-results');
    if (visibleCount === 0 && searchTerm && options.length > 0) {
        if (!noResultsMsg) {
            noResultsMsg = document.createElement('div');
            noResultsMsg.className = 'no-results';
            noResultsMsg.textContent = 'No tasks match your search';
            dropdownMenu.appendChild(noResultsMsg);
        }
        noResultsMsg.style.display = 'block';
    } else if (noResultsMsg) {
        noResultsMsg.style.display = 'none';
    }
}

function selectTask(taskId, taskText) {
    selectedTaskId = taskId;
    dropdownToggle.textContent = taskText || 'All Tasks - Select Task ID to Inspect';
    
    // Update button state
    viewSyncDetailsBtn.disabled = !taskId;
    
    // Update selected state in dropdown
    const options = dropdownMenu.querySelectorAll('.dropdown-option');
    options.forEach(option => {
        if (option.dataset.taskId === taskId) {
            option.classList.add('selected');
        } else {
            option.classList.remove('selected');
        }
    });
    
    // Re-render table to show only selected task or all tasks
    renderTaskHeaders(currentTasks);
    
    closeDropdown();
}

    // Completion buttons
    markCcxBtn.addEventListener('click', function() {
        markStepCompleted('CCX');
    });

    markInforBtn.addEventListener('click', function() {
        markStepCompleted('Infor');
    });

    // Go home button
    goHomeBtn.addEventListener('click', function() {
        window.location.href = '/common/home';
    });

    // Next step button
    nextStepBtn.addEventListener('click', function() {
        proceedToNextStep();
    });


function toggleDropdown() {
    const isOpen = dropdownMenu.classList.contains('show');
    
    if (isOpen) {
        closeDropdown();
    } else {
        openDropdown();
    }
}

function openDropdown() {
    dropdownMenu.classList.add('show');
    dropdownToggle.classList.add('open');
    searchInput.focus();
}

function closeDropdown() {
    dropdownMenu.classList.remove('show');
    dropdownToggle.classList.remove('open');
    searchInput.value = '';
    filterDropdownOptions(); // Reset filter
}

function filterDropdownOptions() {
    const searchTerm = searchInput.value.toLowerCase();
    const options = dropdownMenu.querySelectorAll('.dropdown-option');
    let visibleCount = 0;
    
    options.forEach(option => {
        const text = option.textContent.toLowerCase();
        const shouldShow = text.includes(searchTerm);
        
        if (shouldShow) {
            option.classList.remove('hidden');
            visibleCount++;
        } else {
            option.classList.add('hidden');
        }
    });
    
    // Show/hide no results message
    let noResultsMsg = dropdownMenu.querySelector('.no-results');
    if (visibleCount === 0 && searchTerm && options.length > 0) {
        if (!noResultsMsg) {
            noResultsMsg = document.createElement('div');
            noResultsMsg.className = 'no-results';
            noResultsMsg.textContent = 'No tasks match your search';
            dropdownMenu.appendChild(noResultsMsg);
        }
        noResultsMsg.style.display = 'block';
    } else if (noResultsMsg) {
        noResultsMsg.style.display = 'none';
    }
}

function selectTask(taskId, taskText) {
    selectedTaskId = taskId;
    dropdownToggle.textContent = taskText || 'All Tasks - Select Task ID to Inspect';
    
    // Update button state
    viewSyncDetailsBtn.disabled = !taskId;
    
    // Update selected state in dropdown
    const options = dropdownMenu.querySelectorAll('.dropdown-option');
    options.forEach(option => {
        if (option.dataset.taskId === taskId) {
            option.classList.add('selected');
        } else {
            option.classList.remove('selected');
        }
    });
    
    // Re-render table to show only selected task or all tasks
    renderTaskHeaders(currentTasks);
    
    closeDropdown();
}

function showLoading() {
    loadingSpinner.style.display = 'flex';
}

function hideLoading() {
    loadingSpinner.style.display = 'none';
}

function showError(message) {
    // Create and show error alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.innerHTML = `
        <strong>Error:</strong> ${message}
        <button type="button" class="close" data-dismiss="alert" aria-label="Close">
            <span aria-hidden="true">&times;</span>
        </button>
    `;
    
    // Insert at the beginning of step-content
    const stepContent = document.querySelector('.step-content');
    stepContent.insertBefore(alertDiv, stepContent.firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function showSuccess(message) {
    // Create and show success alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show';
    alertDiv.innerHTML = `
        <strong>Success:</strong> ${message}
        <button type="button" class="close" data-dismiss="alert" aria-label="Close">
            <span aria-hidden="true">&times;</span>
        </button>
    `;
    
    // Insert at the beginning of step-content
    const stepContent = document.querySelector('.step-content');
    stepContent.insertBefore(alertDiv, stepContent.firstChild);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 3000);
}

async function loadTaskHeaders() {
    try {
        showLoading();
        
        const response = await fetch(getApiUrl('/data-synchronization/task-headers'));
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.message || 'Failed to load task headers');
        }
        
        currentTasks = data.tasks;
        renderTaskHeaders(currentTasks);
        populateTaskFilter(currentTasks);
        
    } catch (error) {
        console.error('Error loading task headers:', error);
        showError('Failed to load task headers: ' + error.message);
    } finally {
        hideLoading();
    }
}

function renderTaskHeaders(tasks) {
    const filteredTasks = getFilteredTasks(tasks);
    
    // Update count information
    updateTaskCountInfo(filteredTasks.length, tasks.length);
    
    taskHeaderTbody.innerHTML = '';
    
    if (filteredTasks.length === 0) {
        const noDataMessage = 'No tasks available for synchronization inspection';
        
        taskHeaderTbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center text-muted">${noDataMessage}</td>
            </tr>
        `;
        return;
    }
    
    filteredTasks.forEach(task => {
        const row = document.createElement('tr');
        
        // Format status badges
        const statusBadge = getStatusBadge(task.Status);
        const status2Badge = getStatusBadge(task.Status2);
        const errorBadge = (task.WithError === 'Yes' || task.WithError === true) ? 
            '<span class="status-badge status-error">Yes</span>' : 
            '<span class="status-badge status-completed">No</span>';
        
        row.innerHTML = `
            <td title="${task.TaskID}">${task.TaskID}</td>
            <td title="${task.UserID}">${task.UserID}</td>
            <td title="${task.TPFileName}">${task.TPFileName}</td>
            <td>${statusBadge}</td>
            <td>${status2Badge}</td>
            <td>${errorBadge}</td>
        `;
        
        // Highlight if this is the selected task
        if (selectedTaskId && task.TaskID === selectedTaskId) {
            row.classList.add('table-active');
        }
        
        taskHeaderTbody.appendChild(row);
    });
}

function updateTaskCountInfo(filteredCount, totalCount) {
    if (selectedTaskId) {
        taskCountInfo.textContent = `Showing selected task: ${selectedTaskId}`;
    } else {
        taskCountInfo.textContent = `Showing all ${totalCount} tasks`;
    }
}

function populateTaskFilter(tasks) {
    // Clear existing options
    const existingOptions = dropdownMenu.querySelectorAll('.dropdown-option');
    existingOptions.forEach(option => option.remove());
    
    // Add "All Tasks" option
    const allTasksOption = document.createElement('div');
    allTasksOption.className = 'dropdown-option';
    allTasksOption.dataset.taskId = '';
    allTasksOption.textContent = 'All Tasks - Show All';
    allTasksOption.addEventListener('click', function() {
        selectTask('', 'All Tasks - Select Task ID to Inspect');
    });
    dropdownMenu.appendChild(allTasksOption);
    
    // Add task options
    tasks.forEach(task => {
        const option = document.createElement('div');
        option.className = 'dropdown-option';
        option.dataset.taskId = task.TaskID;
        option.textContent = `${task.TaskID} - ${truncateText(task.TPFileName, 50)}`;
        
        option.addEventListener('click', function() {
            selectTask(task.TaskID, this.textContent);
        });
        
        dropdownMenu.appendChild(option);
    });
}

function getFilteredTasks(tasks) {
    // If a specific task is selected, return only that task
    if (selectedTaskId) {
        return tasks.filter(task => task.TaskID === selectedTaskId);
    }
    
    // Return all tasks when no specific task is selected
    return tasks;
}

function getStatusBadge(status) {
    const statusClass = {
        'Pending': 'status-pending',
        'Completed': 'status-completed',
        'Error': 'status-error',
        'Hold': 'status-hold'
    };
    
    const className = statusClass[status] || 'status-pending';
    return `<span class="status-badge ${className}">${status || 'Unknown'}</span>`;
}

function truncateText(text, maxLength) {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

async function loadSyncDetails(taskId) {
    try {
        showLoading();
        
        const response = await fetch(getApiUrl(`/data-synchronization/sync-details/${taskId}`));
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.message || 'Failed to load sync details');
        }
        
        renderSyncDetails(data.data);
        
    } catch (error) {
        console.error('Error loading sync details:', error);
        showError('Failed to load sync details: ' + error.message);
    } finally {
        hideLoading();
    }
}

function renderSyncDetails(syncData) {
    // Update selected task ID
    selectedTaskIdSpan.textContent = syncData.taskId;
    
    // Update summary statistics
    document.getElementById('total-tp-items').textContent = syncData.summary.totalTPItems;
    document.getElementById('total-changed-items').textContent = syncData.summary.totalChangedItems;
    document.getElementById('total-error-items').textContent = syncData.summary.totalErrorItems;
    
    // Check if there are errors
    hasErrors = syncData.summary.totalErrorItems > 0;
    
    // Render contracts affected table
    renderContractsAffected(syncData.contractsAffected);
    
    // Render TP rows status table
    renderTPRowsStatus(syncData.tpRowsStatus);
    
    // Render exported rows status table
    renderExportedRowsStatus(syncData.exportedRowsStatus);
    
    // Show sync details section and completion section
    syncDetailsSection.style.display = 'block';
    completionSection.style.display = 'block';
    
    // Update comment section visibility based on errors
    updateCommentSection();
}

function renderContractsAffected(contracts) {
    const tbody = document.getElementById('contracts-affected-tbody');
    tbody.innerHTML = '';
    
    if (contracts.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="4" class="text-center text-muted">No contracts affected</td>
            </tr>
        `;
        return;
    }
    
    contracts.forEach(contract => {
        const row = document.createElement('tr');
        const statusBadge = getStatusBadge(contract.status);
        
        row.innerHTML = `
            <td>${contract.contractNumber}</td>
            <td>${contract.vendorId}</td>
            <td class="text-center">${contract.changesCount}</td>
            <td>${statusBadge}</td>
        `;
        
        tbody.appendChild(row);
    });
}

function renderTPRowsStatus(tpRows) {
    const tbody = document.getElementById('tp-rows-tbody');
    tbody.innerHTML = '';
    
    if (tpRows.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="text-center text-muted">No TP rows to display</td>
            </tr>
        `;
        return;
    }
    
    tpRows.forEach(row => {
        const tr = document.createElement('tr');
        const statusBadge = getStatusBadge(row.syncStatus);
        
        tr.innerHTML = `
            <td>${row.itemNumber}</td>
            <td>${row.contractNumber}</td>
            <td>${row.action}</td>
            <td>${statusBadge}</td>
            <td>${formatDateTime(row.lastUpdated)}</td>
        `;
        
        tbody.appendChild(tr);
    });
}

function renderExportedRowsStatus(exportedRows) {
    const tbody = document.getElementById('exported-rows-tbody');
    tbody.innerHTML = '';
    
    if (exportedRows.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="text-center text-muted">No exported rows to display</td>
            </tr>
        `;
        return;
    }
    
    exportedRows.forEach(row => {
        const tr = document.createElement('tr');
        const statusBadge = getStatusBadge(row.exportStatus);
        
        tr.innerHTML = `
            <td>${row.exportGroup}</td>
            <td>${row.contractNumber}</td>
            <td class="text-center">${row.itemsCount}</td>
            <td>${statusBadge}</td>
            <td>${formatDateTime(row.exportDate)}</td>
        `;
        
        tbody.appendChild(tr);
    });
}

function formatDateTime(dateString) {
    if (!dateString) return 'N/A';
    try {
        const date = new Date(dateString);
        return date.toLocaleString();
    } catch {
        return dateString;
    }
}

function updateCommentSection() {
    if (hasErrors) {
        commentSection.style.display = 'block';
        completionComment.required = true;
    } else {
        commentSection.style.display = 'none';
        completionComment.required = false;
        completionComment.value = '';
    }
}

async function markStepCompleted(completionType) {
    // Validate comment if errors exist
    if (hasErrors && !completionComment.value.trim()) {
        showError('A comment is required when errors exist.');
        completionComment.focus();
        return;
    }
    
    try {
        showLoading();
        
        const requestData = {
            taskId: selectedTaskId,
            completionType: completionType,
            comment: completionComment.value.trim()
        };
        
        const response = await fetch(getApiUrl('/data-synchronization/mark-completed'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.message || 'Failed to mark step as completed');
        }
        
        stepCompleted = true;
        nextStepBtn.disabled = false;
        
        // Disable completion buttons
        markCcxBtn.disabled = true;
        markInforBtn.disabled = true;
        
        showSuccess(`Step completed for ${completionType} sync`);
        
    } catch (error) {
        console.error('Error marking step as completed:', error);
        showError('Failed to mark step as completed: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function proceedToNextStep() {
    if (!stepCompleted) {
        showError('Please mark the step as completed before proceeding.');
        return;
    }
    
    try {
        showLoading();
        window.location.href = '/data-synchronization/next-step';
    } catch (error) {
        console.error('Error proceeding to next step:', error);
        showError('Failed to proceed to next step: ' + error.message);
        hideLoading();
    }
}