// step8.js - enhances completion display timeline & counts

document.addEventListener('DOMContentLoaded', function() {
    // Get task ID from URL parameter or data attribute
    const urlParams = new URLSearchParams(window.location.search);
    const taskId = urlParams.get('task_id') || 
                  document.getElementById('completion-container')?.dataset.taskId;
    
    if (taskId) {
        // Fetch initial data
        fetchCompletionData(taskId);
    } else {
        console.error('No task ID found for completion display');
    }
});


async function fetchCompletionData(taskId) {
    try {
        showLoading();
        const response = await fetch(getApiUrl(`/data-synchronization/completion-display/${taskId}`));
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.message || 'Failed to load completion data');
        }
        
        // Use the data directly from the API response
        const data = result.data;
        initializeCompletionDisplay(taskId, data.timeline, data.counts, data.errors);
        
    } catch (error) {
        console.error('Error loading completion data:', error);
        showAlert('error', 'Failed to load completion data: ' + error.message);
    } finally {
        hideLoading();
    }
}

function showLoading() {
    const loadingSpinner = document.getElementById('loading-spinner');
    if (loadingSpinner) loadingSpinner.style.display = 'flex';
}

function hideLoading() {
    const loadingSpinner = document.getElementById('loading-spinner');
    if (loadingSpinner) loadingSpinner.style.display = 'none';
}

function showAlert(type, message, timeout = 5000) {
    // Use the global alert function from layout.html if available
    if (window && typeof window.showGlobalAlert === 'function') {
        window.showGlobalAlert(type, message, timeout);
    } else {
        // Fallback
        console[type === 'error' ? 'error' : 'log'](message);
    }
}

function initializeCompletionDisplay(taskId, timeline, counts, errors) {
    console.log('Initializing completion display for task:', taskId);
    
    // Update task ID display
    const taskIdDisplay = document.getElementById('task-id-display');
    if (taskIdDisplay) taskIdDisplay.textContent = taskId;
    
    // Update timeline display
    updateTimelineDisplay(timeline);
    
    // Update counts display
    updateCountsDisplay(counts);
    
    // Handle any errors
    handleDisplayErrors(errors);
    
    // Add interactive features
    addInteractiveFeatures(taskId);
    
    // Refresh data periodically if needed
    setupDataRefresh(taskId);
}

// Implement the missing functions
function updateTimelineDisplay(timeline) {
    if (!timeline || !Array.isArray(timeline)) return;

    // First update labels & completion state
    timeline.forEach(item => {
        const stepElement = document.querySelector(`.timeline-step[data-key="${item.key}"]`);
        if (!stepElement) return;
        const tsElement = stepElement.querySelector('.ts');
        if (tsElement) {
            if (item.display) {
                // Attempt to split date/time if recognizable
                const parts = item.display.split(/\s+/);
                if (parts.length >= 2) {
                    const datePart = parts[0];
                    const timePart = parts.slice(1).join(' ');
                    tsElement.innerHTML = `<span>${datePart}</span><br><span>${timePart}</span>`;
                } else {
                    tsElement.textContent = item.display;
                }
                tsElement.classList.remove('missing');
            } else { tsElement.textContent = 'Not available'; tsElement.classList.add('missing'); }
        }
        if (item.timestamp) stepElement.classList.add('completed'); else stepElement.classList.remove('completed');
    });

    // Compute proportional positions (created -> completed) if at least two timestamps
    computeProportionalPositions(timeline);

    // After positioning, update fixed exported label & connector
    updateExportedFixedLabel(timeline);
}

function computeProportionalPositions(timeline) {
    const steps = timeline.filter(t => t.timestamp).sort((a,b)=> new Date(a.timestamp) - new Date(b.timestamp));
    const created = timeline.find(t => t.key === 'created');
    const completed = timeline.find(t => t.key === 'completed');
    const axis = document.querySelector('.timeline-axis');
    const progressFill = document.getElementById('timeline-progress-fill');
    if (!axis) return;

    let fallback = false;
    if (!created || !completed || !created.timestamp || !completed.timestamp) fallback = true;
    const start = created?.timestamp ? new Date(created.timestamp).getTime() : null;
    const end = completed?.timestamp ? new Date(completed.timestamp).getTime() : null;
    if (!start || !end || end - start <= 0) fallback = true;

    // When fallback: distribute evenly within 80% width (10% to 90%)
    if (fallback) {
        const allSteps = document.querySelectorAll('.timeline-step');
        const n = allSteps.length - 1;
        allSteps.forEach((el,i)=>{ 
            const position = 10 + (i / n * 80); // Scale from 10% to 90%
            el.style.left = position + '%'; 
        });
        if (progressFill) {
            const completedCount = timeline.filter(t=> t.timestamp).length;
            // Map completed count to full axis percentage (0..100)
            const pct = (completedCount - 1) / (timeline.length -1) * 100;
            progressFill.style.width = Math.max(0, Math.min(100, pct)) + '%';
        }
        return;
    }

    // Position each known timestamp proportionally between start and end (scaled to 80% width)
    timeline.forEach(item => {
        const el = document.querySelector(`.timeline-step[data-key="${item.key}"]`);
        if (!el) return;
        if (item.timestamp) {
            const pos = (new Date(item.timestamp).getTime() - start) / (end - start);
            const scaledPos = 10 + (pos * 80); // Scale from 10% to 90%
            el.style.left = scaledPos + '%';
        }
    });

    // If some timestamps missing (e.g. exported), keep its existing left unless between known neighbors
    // Optionally we could interpolate; for now leave as-is when no timestamp.

    // Progress fill extends to latest completed timestamp (scaled to 80% width)
    if (progressFill) {
        const completedWithTime = timeline.filter(t=> t.timestamp).map(t=> new Date(t.timestamp).getTime());
        const latest = Math.max(...completedWithTime);
        // Progress fill is relative to the full axis (0..100). Keep steps visually within 10..90 but fill uses full axis.
        const pct = (latest - start) / (end - start) * 100;
        progressFill.style.width = Math.max(0, Math.min(100, pct)) + '%';
    }
}

function updateExportedFixedLabel(timeline) {
    const exportedData = timeline.find(t => t.key === 'exported');
    const fixedLabel = document.getElementById('exported-fixed-label');
    const fixedTs = document.getElementById('exported-fixed-ts');
    
    if (!fixedLabel || !fixedTs) return;

    // Update timestamp content
    if (exportedData && exportedData.display) {
        const parts = exportedData.display.split(/\s+/);
        if (parts.length >= 2) {
            const datePart = parts[0];
            const timePart = parts.slice(1).join(' ');
            fixedTs.innerHTML = `<span>${datePart}</span><br><span>${timePart}</span>`;
            fixedTs.classList.remove('missing');
        } else {
            fixedTs.textContent = exportedData.display;
            fixedTs.classList.remove('missing');
        }
    } else {
        fixedTs.textContent = 'Not available';
        fixedTs.classList.add('missing');
    }

    // Update connector line using the dedicated function
    updateExportedConnector();
}

function updateCountsDisplay(counts) {
    if (!counts) return;
    // Update processed items
    const processedItemsElement = document.getElementById('processed-items-value');
    if (processedItemsElement && counts.processed_items !== undefined) {
        animateCountUpdate(processedItemsElement, counts.processed_items);
    }
    
    // Update affected items
    const affectedItemsElement = document.getElementById('affected-items-value');
    if (affectedItemsElement && counts.affected_items !== undefined) {
        animateCountUpdate(affectedItemsElement, counts.affected_items);
    }
}

function animateCountUpdate(element, newValue) {
    // Parse current value
    const currentValue = parseInt(element.textContent) || 0;
    const targetValue = parseInt(newValue) || 0;
    
    if (currentValue === targetValue) return;
    
    // Determine if counting up or down
    const isCountingUp = targetValue > currentValue;
    
    // Set duration based on difference
    const difference = Math.abs(targetValue - currentValue);
    const duration = Math.min(1000, Math.max(500, difference * 10));
    
    // Start time
    const startTime = performance.now();
    
    // Animation function
    function updateCount(timestamp) {
        const elapsed = timestamp - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease-out)
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        
        // Calculate current count
        const currentCount = isCountingUp 
            ? Math.floor(currentValue + (targetValue - currentValue) * easeProgress)
            : Math.floor(currentValue - (currentValue - targetValue) * easeProgress);
        
        // Update display
        element.textContent = currentCount.toLocaleString();
        
        // Continue animation if not complete
        if (progress < 1) {
            requestAnimationFrame(updateCount);
        } else {
            // Ensure final value is set exactly
            element.textContent = targetValue.toLocaleString();
        }
    }
    
    // Start animation
    requestAnimationFrame(updateCount);
}

function handleDisplayErrors(errors) {
    if (!errors) return;
    
    // Check for timestamp errors
    if (errors.timestamps) {
        console.warn('Timeline error:', errors.timestamps);
        showAlert('warning', 'Some timeline data could not be loaded properly.');
    }
    
    // Check for count errors
    if (errors.counts) {
        console.warn('Counts error:', errors.counts);
        showAlert('warning', 'Processing counts data could not be loaded properly.');
    }
}

function addInteractiveFeatures(taskId) {
    // Example: Add tooltips to timeline steps
    const timelineSteps = document.querySelectorAll('.timeline-step');
    timelineSteps.forEach(step => {
        // Add hover effects or click handlers if needed
    });
}

function setupDataRefresh(taskId) {
    // Check if auto-refresh is needed (e.g., if completion is pending)
    const allCompleted = document.querySelectorAll('.timeline-step.completed').length === 
                        document.querySelectorAll('.timeline-step').length;
    
    if (!allCompleted) {
        // Refresh data every 30 seconds if not all steps are completed
        const refreshInterval = setInterval(() => {
            refreshCompletionData(taskId);
            
            // Check again if all complete after refresh
            const allNowCompleted = document.querySelectorAll('.timeline-step.completed').length === 
                                   document.querySelectorAll('.timeline-step').length;
            
            if (allNowCompleted) {
                clearInterval(refreshInterval);
            }
        }, 30000);
    }
}

// Modified refreshCompletionData function to use JSON response
function refreshCompletionData(taskId) {
    // Fetch updated data from the server
    fetch(getApiUrl(`/data-synchronization/completion-display/${taskId}`))
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(result => {
            if (!result.success) {
                throw new Error(result.message || 'Failed to refresh completion data');
            }
            
            const data = result.data;
            
            // Update the display with new data
            updateTimelineDisplay(data.timeline);
            updateCountsDisplay(data.counts);
            handleDisplayErrors(data.errors);
        })
        .catch(error => {
            console.error('Error refreshing completion data:', error);
        });
}

// Debounced resize handler to recalculate connector position
let resizeTimeout;
function handleResize() {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        // Find current timeline data and update connector
        const exportedStep = document.querySelector('.timeline-step[data-key="exported"]');
        if (exportedStep) {
            // Get current timeline data from DOM or stored data
            updateExportedConnector();
        }
    }, 150); // Debounce resize events
}

// Function to update just the connector line without changing content
function updateExportedConnector() {
    const fixedLabel = document.getElementById('exported-fixed-label');
    const fixedTs = document.getElementById('exported-fixed-ts');
    const exportedStep = document.querySelector('.timeline-step[data-key="exported"]');
    const connector = document.getElementById('exported-connector');
    const overlay = document.getElementById('timeline-overlay');
    
    if (!fixedLabel || !exportedStep || !connector || !overlay) return;

    // Draw connector line from fixed label bottom center to exported dot center
    requestAnimationFrame(() => {
        const labelRect = fixedLabel.getBoundingClientRect();
        const dot = exportedStep.querySelector('.dot');
        if (!dot) return;
        
        const dotRect = dot.getBoundingClientRect();
        const overlayRect = overlay.getBoundingClientRect();
        
        const x1 = labelRect.left + labelRect.width/2 - overlayRect.left;
        const y1 = labelRect.top - overlayRect.top; // start from top of label (since label is below dot)
        const x2 = dotRect.left + dotRect.width/2 - overlayRect.left;
        const y2 = dotRect.top + dotRect.height/2 - overlayRect.top;
        
        connector.setAttribute('x1', x1);
        connector.setAttribute('y1', y1);
        connector.setAttribute('x2', x2);
        connector.setAttribute('y2', y2);
    });
}

// Add resize event listener
window.addEventListener('resize', handleResize);

// Export functions for potential external use
window.Step8Display = {
    updateTimeline: updateTimelineDisplay,
    updateCounts: updateCountsDisplay,
    refresh: refreshCompletionData,
    load: fetchCompletionData
};