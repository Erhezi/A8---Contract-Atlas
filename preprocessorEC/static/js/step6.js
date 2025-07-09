document.addEventListener('DOMContentLoaded', function() {
    console.log('Step 6 - Export Changes initialized');
    
    // Add any specific functionality for step 6 here
    
    // Example: Add form submission handling if needed
    const exportForm = document.querySelector('.export-options form');
    if (exportForm) {
        exportForm.addEventListener('submit', function(event) {
            // You can add pre-submission validation or handling here if needed
            console.log('Export format selected:', document.getElementById('export_format').value);
        });
    }
});