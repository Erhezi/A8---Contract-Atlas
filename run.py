# Main application module for the preprocessorEC package
# This serves as the entry point for running the Flask application

from preprocessorEC import create_app
from waitress import serve

# Create the application using our factory function
app = create_app()

if __name__ == '__main__':
    # print("Starting Flask application with restructured architecture...")
    # app.run(debug=True)

    print("Starting Waitress server...")
    serve(app, host='0.0.0.0', port=8090)