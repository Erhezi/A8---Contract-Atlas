document.addEventListener('DOMContentLoaded', function() {
    // Cache DOM elements
    const goToExportBtn = document.getElementById('go-to-export-btn');
    const goHomeBtn = document.getElementById('go-home-btn');
    const deleteTaskBtns = document.querySelectorAll('.delete-task-btn');
    const loadingSpinner = document.getElementById('loading-spinner');

    // Navigate to Export Function (Step 6)
    if (goToExportBtn) {
        goToExportBtn.addEventListener('click', function() {
            if (loadingSpinner) {
                loadingSpinner.style.display = 'flex';
            }
            window.location.href = getApiUrl('/goto-step/6');
        });
    }

    // Navigate to Home
    if (goHomeBtn) {
        goHomeBtn.addEventListener('click', function() {
            if (loadingSpinner) {
                loadingSpinner.style.display = 'flex';
            }
            window.location.href = getApiUrl('/');
        });
    }

    // Confirm delete task
    if (deleteTaskBtns) {
        deleteTaskBtns.forEach(function(btn) {
            btn.addEventListener('click', function(event) {
                if (!confirm('Are you sure you want to delete this task?')) {
                    event.preventDefault();
                    return false;
                }
                if (loadingSpinner) {
                    loadingSpinner.style.display = 'flex';
                }
                return true;
            });
        });
    }

    // Helper function to get API URL (consistent with other JS files)
    function getApiUrl(path) {
        return path;
    }
});