/*step7*/
/** * Step 7: Data Synchronization JavaScript
 * Handles task selection, synchronization details loading, TP and Exported rows rendering,
 * view toggles, and step completion.
*/
// Global variables
let currentTasks = [];
let selectedTaskId = null;
let hasErrors = false;
let stepCompleted = false;
// Hold latest TP rows dataset for rollup computations
let currentTPRowsStatus = [];
// Current TP rows view mode: 'unsynced' | 'synced' | 'rollup'
let currentTPViewMode = 'unsynced';
// Hold latest Exported rows dataset and view mode
let currentExportedRowsStatus = [];
let currentExportedViewMode = 'unsynced'; // 'unsynced' | 'synced' | 'rollup'

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

    //check if a taskID is passed in the URL
    const urlParams = new URLSearchParams(window.location.search);
    const taskIdFromUrl = urlParams.get('task_id');
    if (taskIdFromUrl) {
        console.log('Task ID from URL:', taskIdFromUrl);
        selectTask(taskIdFromUrl, `Task ID: ${taskIdFromUrl}`);
    }

    setupEventListeners();
    updateTpSubheadTop();
    updateExportedSubheadTop();
    
    const tpUnsyncedBtn = document.getElementById('tp-view-unsynced-btn');
    const tpSyncedBtn = document.getElementById('tp-view-synced-btn');
    const tpRollupBtn = document.getElementById('tp-view-rollup-btn');
    if (tpUnsyncedBtn) tpUnsyncedBtn.addEventListener('click', () => switchTPView('unsynced'));
    if (tpSyncedBtn) tpSyncedBtn.addEventListener('click', () => switchTPView('synced'));
    if (tpRollupBtn) tpRollupBtn.addEventListener('click', () => switchTPView('rollup'));
    
    // Exported rows unsynced/synced toggle if available
    const expUnsyncedBtn = document.getElementById('exported-view-unsynced-btn');
    const expSyncedBtn = document.getElementById('exported-view-synced-btn');
    const expRollupBtn = document.getElementById('exported-view-rollup-btn');
    if (expUnsyncedBtn) expUnsyncedBtn.addEventListener('click', () => switchExportedView('unsynced'));
    if (expSyncedBtn) expSyncedBtn.addEventListener('click', () => switchExportedView('synced'));
    if (expRollupBtn) expRollupBtn.addEventListener('click', () => switchExportedView('rollup'));
});

window.addEventListener('resize', updateTpSubheadTop);
window.addEventListener('resize', updateExportedSubheadTop);

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

function createCustomDropdown() {
    const wrapper = document.querySelector('.custom-select-wrapper');
    
    // Create custom dropdown structure
    customDropdown = document.createElement('div');
    customDropdown.className = 'custom-dropdown';
    
    dropdownToggle = document.createElement('button');
    dropdownToggle.className = 'custom-dropdown-toggle';
    dropdownToggle.textContent = 'All Tasks - Select Task ID to Inspect (only exported tasks will be shown)';
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
    dropdownToggle.textContent = taskText || 'All Tasks - Select Task ID to Inspect (only exported tasks will be shown)';
    
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


function showAlert(type, message, timeout = 5000) {
    // Use the global alert function from layout.html
    if (window && typeof window.showGlobalAlert === 'function') {
        window.showGlobalAlert(type, message, timeout);
    } else {
        // Fallback (dev only)
        console[type === 'error' ? 'error' : 'log'](message);
    }
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
        showAlert('error', 'Failed to load task headers: ' + error.message);
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
    allTasksOption.textContent = 'All Tasks - Show All Available to Inspect';
    allTasksOption.addEventListener('click', function() {
        selectTask('', 'All Tasks - Select Task ID to Inspect (only exported tasks will be shown)');
    });
    dropdownMenu.appendChild(allTasksOption);
    
    // Add task options
    tasks.forEach(task => {
        const option = document.createElement('div');
        option.className = 'dropdown-option';
        option.dataset.taskId = task.TaskID;
        option.textContent = `${task.TaskID} - ${task.UserID} - ${truncateText(task.TPFileName, 50)}`;
        
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
        showAlert('error', 'Failed to load sync details: ' + error.message);
    } finally {
        hideLoading();
    }
}

function updateTpSubheadTop() {
    const table = document.getElementById('tp-rows-table');
    if (!table) return;
    const groupRow = table.querySelector('thead .head-group');
    if (!groupRow) return;
    const h = groupRow.getBoundingClientRect().height;
    table.style.setProperty('--tp-subhead-top', `${Math.ceil(h)}px`);
}

function updateExportedSubheadTop() {
    const table = document.getElementById('exported-rows-table');
    if (!table) return;
    const groupRow = table.querySelector('thead .head-group');
    if (!groupRow) return;
    const h = groupRow.getBoundingClientRect().height;
    table.style.setProperty('--exported-subhead-top', `${Math.ceil(h)}px`);
}

function renderSyncDetails(syncData) {
    // Make the section visible early to avoid staying hidden if optional UI fields are missing
    if (syncDetailsSection) syncDetailsSection.style.display = 'block';
    if (completionSection) completionSection.style.display = 'block';

    // Update selected task ID
    if (selectedTaskIdSpan) selectedTaskIdSpan.textContent = syncData.taskId;

    // Helper to safely set text when element exists
    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };

    // Update summary statistics (guard elements that may not exist yet)
    setText('total-tp-items', syncData.summary.totalTPItems);
    setText('total-tp-items-execute', syncData.summary.totalTPItemsExecute);
    setText('total-tp-items-pending', syncData.summary.totalTPItemsPending);

    setText('total-changed-items', syncData.summary.totalChangedItems);
    // These sub-counters may not be in the template yet
    setText('total-changed-items-execute', syncData.summary.totalChangedItemsExecute);
    setText('total-changed-items-pending', syncData.summary.totalChangedItemsPending);

    setText('total-error-items', syncData.summary.totalErrorItems);
    setText('total-error-items-ccx', syncData.summary.totalErrorItemsCCX);
    setText('total-error-items-tp', syncData.summary.totalErrorItemsTP);
    
    // Check if there are errors (handle 'NA' values)
    const errorCount = syncData.summary.totalErrorItems;
    hasErrors = errorCount !== 'NA' && errorCount > 0;
    
    // Render contracts affected table
    renderContractsAffected(syncData.contractsAffected);
    
    // Store dataset, prepare rollup, and default to unsynced view
    currentTPRowsStatus = syncData.tpRowsStatus || [];
    // Update header row counts
    updateTPHeaderCounts(currentTPRowsStatus);
    renderTPRowsRollup(currentTPRowsStatus);
    // Render immediately (detailed) so user sees data even if toggle controls are missing
    renderTPRowsStatus(currentTPRowsStatus);
    // Then apply the unsynced view filter/toggles if available
    switchTPView('unsynced');
    
    // Render exported rows status table using real dataset
    currentExportedRowsStatus = Array.isArray(syncData.exportedRowsStatus) ? syncData.exportedRowsStatus : [];
    // If exported rows container/toggles are missing, try to noop; renderer will use existing DOM if present
    updateExportedHeaderCounts(currentExportedRowsStatus);
    renderExportedRowsStatus(currentExportedRowsStatus);
    renderExportedRowsRollup(currentExportedRowsStatus);
    // Default to unsynced view for exported
    switchExportedView('unsynced');
    
    // Update comment section visibility based on errors
    updateCommentSection();
}

// Show/hide the comment box when errors exist
function updateCommentSection() {
    if (!commentSection) return;
    // Get unsynced counts from both tables
    const tpUnsyncedCount = parseInt(document.getElementById('tp-rows-unsynced-count')?.textContent || '0', 10);
    const exportedUnsyncedCount = parseInt(document.getElementById('exported-rows-unsynced-count')?.textContent || '0', 10);

    // Show the comment section if there are errors or unsynced rows in either table
    commentSection.style.display = hasErrors || tpUnsyncedCount > 0 || exportedUnsyncedCount > 0 ? 'block' : 'none';
}

// Update TP header counts (total, unsynced, synced)
function updateTPHeaderCounts(rows) {
    const totalEl = document.getElementById('tp-rows-count');
    const unsyncedEl = document.getElementById('tp-rows-unsynced-count');
    const syncedEl = document.getElementById('tp-rows-synced-count');
    if (!totalEl || !unsyncedEl || !syncedEl) return;
    const isSynced = v => ['1','synced','yes','y'].includes(String(v || '').toLowerCase());
    const data = Array.isArray(rows) ? rows : [];
    let synced = 0;
    let unsynced = 0;
    for (const r of data) {
        const ccx = String(r['TP_CCX_Synced'] || '').toLowerCase();
        const infor = String(r['TP_Infor_Synced'] || '').toLowerCase();
        if (isSynced(ccx) && isSynced(infor)) synced++; else unsynced++;
    }
    totalEl.textContent = data.length;
    unsyncedEl.textContent = unsynced;
    syncedEl.textContent = synced;
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
        
        row.innerHTML = `
            <td title="${contract.contractNumber || 'N/A'}">${truncateText(contract.contractNumber || 'N/A', 20)}</td>
            <td class="text-center">${contract.totalCommittedOperations || 0}</td>
            <td class="text-center">${contract.executableOperations || 0}</td>
            <td class="text-center">${contract.nonExecutableOperations || 0}</td>
        `;
        
        tbody.appendChild(row);
    });
}

function renderTPRowsStatus(tpRows) {
    const tbody = document.getElementById('tp-rows-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!tpRows || tpRows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="26" class="text-center text-muted">No TP rows found for this task</td></tr>`;
        updateTpSubheadTop();
        return;
    }

    const toCell = (v) => (v === null || v === undefined || v === '' ? '' : v);
    const isPositive = (v) => {
        if (v === null || v === undefined || v === -1) return null;
        if (typeof v === 'number') return v > 0 ? 1 : 0;
        const s = String(v).trim().toLowerCase();
        if (["1", "matched", "synced"].includes(s)) return 1;
        if (["0", "no match", "not synced"].includes(s)) return 0;
        return null;
    };

    // Tooltip for Item Matched
    const blueMatchIndicator = (shape, value) => {
        const status = isPositive(value);
        const shapeCls = shape === 'circle' ? 'sync-circle' : 'sync-square';
        const title = status === 1 ? 'Matched' : (status === 0 ? 'Not Matched' : '');
        if (status === 1) return `<span class="${shapeCls} sync-blue" title="${title}"></span>`;
        if (status === 0) return `<span class="${shapeCls} sync-blue-hollow" title="${title}"></span>`;
        return `<span class="${shapeCls} sync-na" title="${title}"></span>`;
    };

    // Tooltip for TP Synced (Overall)
    const tpSyncedIndicator = (shape, value, greyOut = false) => {
        const status = isPositive(value);
        const shapeCls = shape === 'circle' ? 'sync-circle' : 'sync-square';
        const title = status === 1 ? 'Synced' : (status === 0 ? 'Not Synced' : '');
        const cls = greyOut ? 'sync-na' : (status === 1 ? 'sync-ok' : (status === 0 ? 'sync-bad' : 'sync-na'));
        return `<span class="${shapeCls} ${cls}" title="${title}"></span>`;
    };

    // Tooltip for sub-comparisons (QOE, Price, etc.)
    // Format helpers
    function formatInt(val) {
        if (val === null || val === undefined || val === '') return '';
        return parseInt(val, 10);
    }
    function formatMoney(val) {
        if (val === null || val === undefined || val === '') return '';
        let num = Number(val);
        if (isNaN(num)) return val;
        return `$${num.toFixed(2)}`;
    }
    function formatDate(val) {
        if (!val) return '';
        let d = new Date(val);
        if (isNaN(d.getTime())) return val;
        return d.toISOString().slice(0,10);
    }

    const indicatorCell = (shape, value, tpValue, otherValue, otherLabel, greyOut = false, type = 'text') => {
        const status = isPositive(value);
        const cls = greyOut ? 'sync-na' : (status === 1 ? 'sync-ok' : (status === 0 ? 'sync-bad' : 'sync-na'));
        const shapeCls = shape === 'circle' ? 'sync-circle' : 'sync-square';
        let title = '';
        let tpDisplay = tpValue, otherDisplay = otherValue;
        if (type === 'int') {
            tpDisplay = formatInt(tpValue);
            otherDisplay = formatInt(otherValue);
        } else if (type === 'money') {
            tpDisplay = formatMoney(tpValue);
            otherDisplay = formatMoney(otherValue);
        } else if (type === 'date') {
            tpDisplay = formatDate(tpValue);
            otherDisplay = formatDate(otherValue);
        }
        if (otherLabel) {
            title = `TP: ${tpDisplay}\n${otherLabel}: ${otherDisplay}`;
        }
        return `<span class="${shapeCls} ${cls}" title="${title}"></span>`;
    };

    const rowsHtml = tpRows.map(r => {
        const intendedAction = String(r['Intended Action']).trim().toLowerCase();
        const itemMatchedCCX = String(r['Item_Matching_Flag_CCX']).trim().toLowerCase();
        const itemMatchedInfor = String(r['Item_Matching_Flag_Infor']).trim().toLowerCase();
        // Logic for grey out
        const greyOutCCX = intendedAction === 'upsert' && itemMatchedCCX === 'no match';
        const greyOutInfor = intendedAction === 'upsert' && itemMatchedInfor === 'no match';
        const greyOutFromQOE = intendedAction === 'expire';
        return `
        <tr>
            <td title="${toCell(r['Intended Action'])}">${toCell(r['Intended Action'])}</td>
            <td title="${toCell(r['Actual Action'])}">${toCell(r['Actual Action'])}</td>
            <td title="${toCell(r['File Row Action'])}">${toCell(r['File Row Action'])}</td>
            <td title="${toCell(r['Contract Number'])}">${toCell(r['Contract Number'])}</td>
            <td title="${toCell(r['ERP Vendor ID'])}">${toCell(r['ERP Vendor ID'])}</td>
            <td title="${toCell(r['Mfg Part Num'])}">${toCell(r['Mfg Part Num'])}</td>
            <td>${toCell(r['UOM'])}</td>
            <td>${toCell(r['UOM_INFOR'])}</td>
            <td>${toCell(r['Item_TP'])}</td>
            <td>${toCell(r['Item_Infor'])}</td>
            <td>${blueMatchIndicator('square', r['Item_Matching_Flag_CCX'])}</td>
            <td>${blueMatchIndicator('circle', r['Item_Matching_Flag_Infor'])}</td>
            <td>${tpSyncedIndicator('square', r['TP_CCX_Synced'], greyOutCCX)}</td>
            <td>${tpSyncedIndicator('circle', r['TP_Infor_Synced'], greyOutInfor)}</td>
            <td>${indicatorCell('square', r['Match_QOE_TP_CCX'], r['QOE_TP'], r['QOE_CCX'], 'CCX', greyOutCCX || greyOutFromQOE, 'int')}</td>
            <td>${indicatorCell('circle', r['Match_QOE_TP_Infor'], r['QOE_TP'], r['QOE_Infor'], 'Infor', greyOutInfor || greyOutFromQOE, 'int')}</td>
            <td>${indicatorCell('square', r['Match_Price_TP_CCX'], r['Price_TP'], r['Price_CCX'], 'CCX', greyOutCCX || greyOutFromQOE, 'money')}</td>
            <td>${indicatorCell('circle', r['Match_Price_TP_Infor'], r['Price_TP'], r['Price_Infor'], 'Infor', greyOutInfor || greyOutFromQOE, 'money')}</td>
            <td>${indicatorCell('square', r['Match_VendorPartNum_TP_CCX'], r['VendorPartNum_TP'], r['VendorPartNum_CCX'], 'CCX', greyOutCCX || greyOutFromQOE)}</td>
            <td>${indicatorCell('circle', r['Match_VendorPartNum_TP_Infor'], r['VendorPartNum_TP'], r['VendorPartNum_Infor'], 'Infor', greyOutInfor || greyOutFromQOE)}</td>
            <td>${indicatorCell('square', r['Match_EffectiveDate_TP_CCX'], r['EffectiveDate_TP'], r['EffectiveDate_CCX'], 'CCX', greyOutCCX || greyOutFromQOE, 'date')}</td>
            <td>${indicatorCell('circle', r['Match_EffectiveDate_TP_Infor'], r['EffectiveDate_TP'], r['EffectiveDate_Infor'], 'Infor', greyOutInfor || greyOutFromQOE, 'date')}</td>
            <td>${indicatorCell('square', r['Match_ExpirationDate_TP_CCX'], r['ExpirationDate_TP'], r['ExpirationDate_CCX'], 'CCX', greyOutCCX || greyOutFromQOE, 'date')}</td>
            <td>${indicatorCell('circle', r['Match_ExpirationDate_TP_Infor'], r['ExpirationDate_TP'], r['ExpirationDate_Infor'], 'Infor', greyOutInfor || greyOutFromQOE, 'date')}</td>
            <td>${indicatorCell('square', r['Match_Description_TP_CCX'], r['Description_TP'], r['Description_CCX'], 'CCX', greyOutCCX || greyOutFromQOE)}</td>
            <td>${indicatorCell('circle', r['Match_Description_TP_Infor'], r['Description_TP'], r['Description_Infor'], 'Infor', greyOutInfor || greyOutFromQOE)}</td>
        </tr>
        `;
    }).join('');

    tbody.innerHTML = rowsHtml;
    requestAnimationFrame(updateTpSubheadTop);
}

// Minimal no-op to avoid ReferenceError if rollup is not implemented yet
function renderTPRowsRollup(tpRows) {
    const tbody = document.getElementById('tp-rows-rollup-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!tpRows || tpRows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No data available for rollup</td></tr>';
        return;
    }

    const norm = v => (v === null || v === undefined ? '' : String(v).trim());
    const groups = new Map();

    tpRows.forEach(r => {
        const intended = norm(r['Intended Action']);
        const fileRow = norm(r['File Row Action']);
        const key = intended + '||' + fileRow;
        if (!groups.has(key)) {
            groups.set(key, {
                intended,
                fileRow,
                total: 0,
                ccxItemMatched: 0,
                inforItemMatched: 0,
                ccxSynced: 0,
                inforSynced: 0
            });
        }
        const g = groups.get(key);
        g.total += 1;
        const matchCCX = String(r['Item_Matching_Flag_CCX'] || '').toLowerCase();
        const matchInfor = String(r['Item_Matching_Flag_Infor'] || '').toLowerCase();
        if (['1','matched','yes','y'].includes(matchCCX)) g.ccxItemMatched += 1;
        if (['1','matched','yes','y'].includes(matchInfor)) g.inforItemMatched += 1;
        const syncedCCX = String(r['TP_CCX_Synced'] || '').toLowerCase();
        const syncedInfor = String(r['TP_Infor_Synced'] || '').toLowerCase();
        if (['1','synced','yes','y'].includes(syncedCCX)) g.ccxSynced += 1;
        if (['1','synced','yes','y'].includes(syncedInfor)) g.inforSynced += 1;
    });

    const rows = Array.from(groups.values()).sort((a,b)=>{
        if (a.intended === b.intended) return a.fileRow.localeCompare(b.fileRow);
        return a.intended.localeCompare(b.intended);
    });

    const pct = (num, den) => den === 0 ? '0%' : ((num/den)*100).toFixed(1) + '%';

    const html = rows.map(g => `<tr>
        <td>${g.intended || '&nbsp;'}</td>
        <td>${g.fileRow || '&nbsp;'}</td>
        <td class="text-center">${g.total}</td>
        <td class="text-center" title="${g.ccxItemMatched} of ${g.total}">${g.ccxItemMatched} / ${pct(g.ccxItemMatched, g.total)}</td>
        <td class="text-center" title="${g.inforItemMatched} of ${g.total}">${g.inforItemMatched} / ${pct(g.inforItemMatched, g.total)}</td>
        <td class="text-center" title="${g.ccxSynced} of ${g.total}">${g.ccxSynced} / ${pct(g.ccxSynced, g.total)}</td>
        <td class="text-center" title="${g.inforSynced} of ${g.total}">${g.inforSynced} / ${pct(g.inforSynced, g.total)}</td>
    </tr>`).join('');

    tbody.innerHTML = html || '<tr><td colspan="7" class="text-center text-muted">No rollup results</td></tr>';
}


function renderExportedRowsStatus(exportedRows) {
    const tbody = document.getElementById('exported-rows-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!exportedRows || exportedRows.length === 0) {
        tbody.innerHTML = `<tr><td colspan="26" class="text-center text-muted">No Exported rows found for this task</td></tr>`;
        updateTpSubheadTop();
        return;
    }

    // Helpers (mirror TP rows indicators)
    const toCell = (v) => (v === null || v === undefined || v === '' ? '' : v);
    const isPositive = (v) => {
        if (v === null || v === undefined || v === -1) return null;
        if (typeof v === 'number') return v > 0 ? 1 : 0;
        const s = String(v).trim().toLowerCase();
        if (["1", "matched", "synced"].includes(s)) return 1;
        if (["0", "no match", "not synced"].includes(s)) return 0;
        return null;
    };
    // Tooltip for Item Matched
    const blueMatchIndicator = (shape, value) => {
        const status = isPositive(value);
        const shapeCls = shape === 'circle' ? 'sync-circle' : 'sync-square';
        const title = status === 1 ? 'Matched' : (status === 0 ? 'Not Matched' : '');
        if (status === 1) return `<span class="${shapeCls} sync-blue" title="${title}"></span>`;
        if (status === 0) return `<span class="${shapeCls} sync-blue-hollow" title="${title}"></span>`;
        return `<span class="${shapeCls} sync-na" title="${title}"></span>`;
    };
    // Tooltip for TP Synced (Overall)
    const tpSyncedIndicator = (shape, value, greyOut = false) => {
        const status = isPositive(value);
        const shapeCls = shape === 'circle' ? 'sync-circle' : 'sync-square';
        const title = status === 1 ? 'Synced' : (status === 0 ? 'Not Synced' : '');
        const cls = greyOut ? 'sync-na' : (status === 1 ? 'sync-ok' : (status === 0 ? 'sync-bad' : 'sync-na'));
        return `<span class="${shapeCls} ${cls}" title="${title}"></span>`;
    };

    // Tooltip for sub-comparisons (QOE, Price, etc.)
    // Format helpers
    function formatInt(val) {
        if (val === null || val === undefined || val === '') return '';
        return parseInt(val, 10);
    }
    function formatMoney(val) {
        if (val === null || val === undefined || val === '') return '';
        let num = Number(val);
        if (isNaN(num)) return val;
        return `$${num.toFixed(2)}`;
    }
    function formatDate(val) {
        if (!val) return '';
        let d = new Date(val);
        if (isNaN(d.getTime())) return val;
        return d.toISOString().slice(0,10);
    }

    const indicatorCell = (shape, value, tpValue, otherValue, otherLabel, greyOut = false, type = 'text') => {
        const status = isPositive(value);
        const cls = greyOut ? 'sync-na' : (status === 1 ? 'sync-ok' : (status === 0 ? 'sync-bad' : 'sync-na'));
        const shapeCls = shape === 'circle' ? 'sync-circle' : 'sync-square';
        let tpDisplay = tpValue, otherDisplay = otherValue;
        if (type === 'int') {
            tpDisplay = formatInt(tpValue);
            otherDisplay = formatInt(otherValue);
        } else if (type === 'money') {
            tpDisplay = formatMoney(tpValue);
            otherDisplay = formatMoney(otherValue);
        } else if (type === 'date') {
            tpDisplay = formatDate(tpValue);
            otherDisplay = formatDate(otherValue);
        }
        const title = otherLabel ? `TP: ${tpDisplay}\n${otherLabel}: ${otherDisplay}` : '';
        return `<span class="${shapeCls} ${cls}" title="${title}"></span>`;
    };

    const html = exportedRows.map(r => {
        const actualAction = String(r['Actual Action'] || '').trim().toLowerCase();
        const finalAction = String(r['Final Action'] || '').trim().toLowerCase();
        const itemMatchedCCX = String(r['Item_Matching_Flag_CCX'] || '').trim().toLowerCase();
        const itemMatchedInfor = String(r['Item_Matching_Flag_Infor'] || '').trim().toLowerCase();
        // Grey-out rules: if item not to be expired and no match for respective system; also grey out QOE-era if expire action
        const greyOutCCX = itemMatchedCCX === 'no match' && !actualAction === 'expire';
        const greyOutInfor = itemMatchedInfor === 'no match' && !actualAction === 'expire';
        const greyOutFromQOE = actualAction === 'expire' || finalAction == 'masked';

        return `
        <tr>
            <td title="${formatDate(r['Exported Date'])}">${formatDate(r['Exported Date'])}</td>
            <td title="${toCell(r['Actual Action'])}">${toCell(r['Actual Action'])}</td>
            <td title="${toCell(r['Final Action'])}">${toCell(r['Final Action'])}</td>
            <td title="${toCell(r['Contract Number'])}">${toCell(r['Contract Number'])}</td>
            <td title="${toCell(r['ERP Vendor ID'])}">${toCell(r['ERP Vendor ID'])}</td>
            <td title="${toCell(r['Mfg Part Num'])}">${toCell(r['Mfg Part Num'])}</td>
            <td>${toCell(r['UOM'])}</td>
            <td>${toCell(r['UOM_INFOR'])}</td>
            <td>${toCell(r['Item'])}</td>
            <td>${toCell(r['Item_Infor'])}</td>
            <td>${blueMatchIndicator('square', r['Item_Matching_Flag_CCX'])}</td>
            <td>${blueMatchIndicator('circle', r['Item_Matching_Flag_Infor'])}</td>
            <td>${tpSyncedIndicator('square', r['TP_CCX_Synced'], greyOutCCX)}</td>
            <td>${tpSyncedIndicator('circle', r['TP_Infor_Synced'], greyOutInfor)}</td>
            <td>${indicatorCell('square', r['Match_QOE_TP_CCX'], r['QOE_TP'], r['QOE_CCX'], 'CCX', greyOutCCX || greyOutFromQOE, 'int')}</td>
            <td>${indicatorCell('circle', r['Match_QOE_TP_Infor'], r['QOE_TP'], r['QOE_Infor'], 'Infor', greyOutInfor || greyOutFromQOE, 'int')}</td>
            <td>${indicatorCell('square', r['Match_Price_TP_CCX'], r['Price_TP'], r['Price_CCX'], 'CCX', greyOutCCX || greyOutFromQOE, 'money')}</td>
            <td>${indicatorCell('circle', r['Match_Price_TP_Infor'], r['Price_TP'], r['Price_Infor'], 'Infor', greyOutInfor || greyOutFromQOE, 'money')}</td>
            <td>${indicatorCell('square', r['Match_VendorPartNum_TP_CCX'], r['VendorPartNum_TP'], r['VendorPartNum_CCX'], 'CCX', greyOutCCX || greyOutFromQOE)}</td>
            <td>${indicatorCell('circle', r['Match_VendorPartNum_TP_Infor'], r['VendorPartNum_TP'], r['VendorPartNum_Infor'], 'Infor', greyOutInfor || greyOutFromQOE)}</td>
            <td>${indicatorCell('square', r['Match_EffectiveDate_TP_CCX'], r['EffectiveDate_TP'], r['EffectiveDate_CCX'], 'CCX', greyOutCCX || greyOutFromQOE, 'date')}</td>
            <td>${indicatorCell('circle', r['Match_EffectiveDate_TP_Infor'], r['EffectiveDate_TP'], r['EffectiveDate_Infor'], 'Infor', greyOutInfor || greyOutFromQOE, 'date')}</td>
            <td>${indicatorCell('square', r['Match_ExpirationDate_TP_CCX'], r['ExpirationDate_TP'], r['ExpirationDate_CCX'], 'CCX', greyOutCCX || greyOutFromQOE, 'date')}</td>
            <td>${indicatorCell('circle', r['Match_ExpirationDate_TP_Infor'], r['ExpirationDate_TP'], r['ExpirationDate_Infor'], 'Infor', greyOutInfor || greyOutFromQOE, 'date')}</td>
            <td>${indicatorCell('square', r['Match_Description_TP_CCX'], r['Description_TP'], r['Description_CCX'], 'CCX', greyOutCCX || greyOutFromQOE)}</td>
            <td>${indicatorCell('circle', r['Match_Description_TP_Infor'], r['Description_TP'], r['Description_Infor'], 'Infor', greyOutInfor || greyOutFromQOE)}</td>
        </tr>`;
    }).join('');

    tbody.innerHTML = html;
    requestAnimationFrame(updateExportedSubheadTop);
}

// Rollup view for exported rows based on exported rows dataset
function renderExportedRowsRollup(exportedRows) {
    const tbody = document.getElementById('exported-rows-rollup-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!Array.isArray(exportedRows) || exportedRows.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No data available for rollup</td></tr>';
        return;
    }

    function formatDate(val) {
        if (!val) return '';
        const d = new Date(val);
        if (isNaN(d.getTime())) return String(val);
        return d.toISOString().slice(0,10);
    }

    const norm = v => (
        v instanceof Date
            ? formatDate(v)
            : (v === null || v === undefined ? '' : String(v).trim())
    );

    const groups = new Map();

    exportedRows.forEach(r => {
        const actualAction = norm(r['Actual Action']);
        const finalAction = norm(r['Final Action']);
        const key = actualAction + '||' + finalAction;
        if (!groups.has(key)) {
            groups.set(key, {
                actualAction,
                finalAction,
                total: 0,
                ccxItemMatched: 0,
                inforItemMatched: 0,
                ccxSynced: 0,
                inforSynced: 0
            });
        }
        const g = groups.get(key);
        g.total += 1;

        const ccxMatch = String(r['Item_Matching_Flag_CCX'] || '').toLowerCase();
        const inforMatch = String(r['Item_Matching_Flag_Infor'] || '').toLowerCase();
        if (['1','matched','yes','y'].includes(ccxMatch)) g.ccxItemMatched += 1;
        if (['1','matched','yes','y'].includes(inforMatch)) g.inforItemMatched += 1;

        const ccxSynced = String(r['TP_CCX_Synced'] || '').toLowerCase();
        const inforSynced = String(r['TP_Infor_Synced'] || '').toLowerCase();
        if (['1','synced','yes','y','true'].includes(ccxSynced)) g.ccxSynced += 1;
        if (['1','synced','yes','y','true'].includes(inforSynced)) g.inforSynced += 1;
    });

    const pct = (num, den) => den === 0 ? '0%' : ((num/den)*100).toFixed(1) + '%';

    const rows = Array.from(groups.values()).sort((a,b) => {
        if (a.actualAction === b.actualAction) return a.actualAction.localeCompare(b.finalAction);
        return a.actualAction.localeCompare(b.actualAction);
    });

    const html = rows.map(g => `
        <tr>
            <td>${g.actualAction || '&nbsp;'}</td>
            <td>${g.finalAction || '&nbsp;'}</td>
            <td class="text-center">${g.total}</td>
            <td class="text-center" title="${g.ccxItemMatched} of ${g.total}">
                ${g.ccxItemMatched} / ${pct(g.ccxItemMatched, g.total)}
            </td>
            <td class="text-center" title="${g.inforItemMatched} of ${g.total}">
                ${g.inforItemMatched} / ${pct(g.inforItemMatched, g.total)}
            </td>
            <td class="text-center" title="${g.ccxSynced} of ${g.total}">
                ${g.ccxSynced} / ${pct(g.ccxSynced, g.total)}
            </td>
            <td class="text-center" title="${g.inforSynced} of ${g.total}">
                ${g.inforSynced} / ${pct(g.inforSynced, g.total)}
            </td>
        </tr>
    `).join('');

    tbody.innerHTML = html || '<tr><td colspan="7" class="text-center text-muted">No rollup results</td></tr>';
}

function switchExportedView(mode) {
    const detailedTable = document.getElementById('exported-rows-table');
    const rollupTable = document.getElementById('exported-rows-table-rollup');
    const rollupBtn = document.getElementById('exported-view-rollup-btn');
    const unsyncedBtn = document.getElementById('exported-view-unsynced-btn');
    const syncedBtn = document.getElementById('exported-view-synced-btn');
    if (!detailedTable || !rollupTable || !unsyncedBtn || !syncedBtn || !rollupBtn) return;

    currentExportedViewMode = mode;

    // Reset button states if present
    [unsyncedBtn, syncedBtn, rollupBtn].forEach(b => b.classList.remove('active'));

    if (mode === 'rollup') {
        detailedTable.style.display = 'none';
        rollupTable.style.display = '';
        if (rollupBtn) rollupBtn.classList.add('active');
    } else {
        rollupTable.style.display = 'none';
        detailedTable.style.display = '';
        if (mode === 'unsynced') {
            unsyncedBtn.classList.add('active');
            // Filter unsynced rows: either CCX or Infor not synced
            const unsynced = currentExportedRowsStatus.filter(r => {
                const ccx = String(r['TP_CCX_Synced'] || '').toLowerCase();
                const infor = String(r['TP_Infor_Synced'] || '').toLowerCase();
                const isSynced = v => ['1','synced','yes','y'].includes(v);
                return !(isSynced(ccx) && isSynced(infor));
            });
            renderExportedRowsStatus(unsynced);
        } else if (mode === 'synced') {
            syncedBtn.classList.add('active');
            const syncedRows = currentExportedRowsStatus.filter(r => {
                const ccx = String(r['TP_CCX_Synced'] || '').toLowerCase();
                const infor = String(r['TP_Infor_Synced'] || '').toLowerCase();
                const isSynced = v => ['1','synced','yes','y'].includes(v);
                return isSynced(ccx) && isSynced(infor);
            });
            renderExportedRowsStatus(syncedRows);
        } else {
            // fallback to full detailed view if needed in future
            renderExportedRowsStatus(currentExportedRowsStatus);
        }
        // Keep counts up to date (based on full dataset)
        updateExportedHeaderCounts(currentExportedRowsStatus);
    }
}

// Update Exported header counts (total, unsynced, synced)
function updateExportedHeaderCounts(rows) {
    const totalEl = document.getElementById('exported-rows-count');
    const unsyncedEl = document.getElementById('exported-rows-unsynced-count');
    const syncedEl = document.getElementById('exported-rows-synced-count');
    if (!totalEl || !unsyncedEl || !syncedEl) return;
    const isSynced = v => ['1','synced','yes','y','true'].includes(String(v || '').toLowerCase());
    const data = Array.isArray(rows) ? rows : [];
    let synced = 0;
    let unsynced = 0;
    for (const r of data) {
        const ccx = String(r['TP_CCX_Synced'] || '').toLowerCase();
        const infor = String(r['TP_Infor_Synced'] || '').toLowerCase();
        if (isSynced(ccx) && isSynced(infor)) synced++; else unsynced++;
    }
    totalEl.textContent = data.length;
    unsyncedEl.textContent = unsynced;
    syncedEl.textContent = synced;
}

function switchTPView(mode) {
    const detailedTable = document.getElementById('tp-rows-table');
    const rollupTable = document.getElementById('tp-rows-table-rollup');
    const unsyncedBtn = document.getElementById('tp-view-unsynced-btn');
    const syncedBtn = document.getElementById('tp-view-synced-btn');
    const rollupBtn = document.getElementById('tp-view-rollup-btn');
    if (!detailedTable || !rollupTable || !unsyncedBtn || !syncedBtn || !rollupBtn) return;

    currentTPViewMode = mode;

    // Reset button active states
    [unsyncedBtn, syncedBtn, rollupBtn].forEach(b => b.classList.remove('active'));

    if (mode === 'rollup') {
        detailedTable.style.display = 'none';
        rollupTable.style.display = '';
        rollupBtn.classList.add('active');
    } else {
        rollupTable.style.display = 'none';
        detailedTable.style.display = '';
        if (mode === 'unsynced') {
            unsyncedBtn.classList.add('active');
            // Filter unsynced rows: either CCX or Infor not synced
            const unsynced = currentTPRowsStatus.filter(r => {
                const ccx = String(r['TP_CCX_Synced'] || '').toLowerCase();
                const infor = String(r['TP_Infor_Synced'] || '').toLowerCase();
                const isSynced = v => ['1','synced','yes','y'].includes(v);
                return !(isSynced(ccx) && isSynced(infor));
            });
            renderTPRowsStatus(unsynced);
        } else if (mode === 'synced') {
            syncedBtn.classList.add('active');
            const syncedRows = currentTPRowsStatus.filter(r => {
                const ccx = String(r['TP_CCX_Synced'] || '').toLowerCase();
                const infor = String(r['TP_Infor_Synced'] || '').toLowerCase();
                const isSynced = v => ['1','synced','yes','y'].includes(v);
                return isSynced(ccx) && isSynced(infor);
            });
            renderTPRowsStatus(syncedRows);
        } else {
            // fallback to full detailed view if needed in future
            renderTPRowsStatus(currentTPRowsStatus);
        }
    }
    // Keep counts up to date (based on full dataset)
    updateTPHeaderCounts(currentTPRowsStatus);
}