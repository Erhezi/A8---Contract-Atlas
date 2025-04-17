# Main application module for the preprocessorEC package
# This serves as the entry point for running the Flask application

from preprocessorEC import create_app

# Create the application using our factory function
app = create_app()

if __name__ == '__main__':
    print("Starting Flask application with restructured architecture...")
    app.run(debug=True)