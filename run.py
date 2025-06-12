# Main application module for the preprocessorEC package
# This serves as the entry point for running the Flask application

from preprocessorEC import create_app
from waitress import serve
import os

# get envrioment configuration
ENV = os.getenv('FLASK_ENV', 'development')
URL_PREFIX = os.getenv('URL_PREFIX', '/preprocessor' if ENV == 'production' else '')

# Create the application using our factory function
app = create_app(ENV)

if __name__ == '__main__':
    if ENV == 'production':
        print(f"Starting Waitress server in PRODUCTION mode with URL_PREFIX={URL_PREFIX}...")
        serve(app, host='0.0.0.0', port=8090)
    else:
        print(f"Starting Flask development server with URL_PREFIX={URL_PREFIX}...")
        app.run(debug=True)