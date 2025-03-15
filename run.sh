#!/bin/bash

# Create necessary directories
mkdir -p logs
mkdir -p data/models
mkdir -p data/historical
mkdir -p data/forecasts
mkdir -p data/visualizations

# Check if Python virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start the Flask API in the background
echo "Starting Flask API..."
python api.py &
API_PID=$!

# Wait for API to start
echo "Waiting for API to start..."
sleep 5

# Check if API is running
echo "Checking if API is running..."
if curl -s http://localhost:8888/api/health > /dev/null; then
    echo "API is running on port 8888."
    
    # Run the test script to verify the API is working correctly
    echo "Running API tests to verify functionality..."
    python test_api.py
    
    echo ""
    echo "NOTE: The first forecast request may take up to 60 seconds to complete."
    echo "      Subsequent requests will be faster due to caching."
    echo ""
else
    echo "Warning: API does not seem to be running on port 8888."
    echo "Please check the API port in api.py and make sure it matches the port in ForecastComponent.jsx."
    echo "Current port in api.py: $(grep -o "port=[0-9]*" api.py | cut -d= -f2)"
fi

# Check if Next.js example directory exists
if [ -d "nextjs-example" ]; then
    cd nextjs-example
    
    # Ensure the API_BASE_URL is set correctly in ForecastComponent.jsx
    echo "Checking API_BASE_URL in ForecastComponent.jsx..."
    API_PORT=$(grep -o "localhost:[0-9]*" ForecastComponent.jsx | cut -d: -f2)
    echo "Current API port in ForecastComponent.jsx: $API_PORT"
    
    if [ "$API_PORT" != "8888" ]; then
        echo "Updating API_BASE_URL to use port 8888..."
        sed -i '' "s/localhost:[0-9]*/localhost:8888/g" ForecastComponent.jsx
        echo "Updated API_BASE_URL in ForecastComponent.jsx to use port 8888."
    fi
    
    # Install Node.js dependencies if node_modules doesn't exist
    if [ ! -d "node_modules" ]; then
        echo "Installing Node.js dependencies..."
        npm install
    fi
    
    # Start Next.js development server
    echo "Starting Next.js development server..."
    npm run dev
    
    # Store Next.js PID
    NEXTJS_PID=$!
    
    # Go back to root directory
    cd ..
else
    echo "Next.js example directory not found. Only running the API."
fi

# Function to handle script termination
cleanup() {
    echo "Shutting down servers..."
    
    # Kill the API process
    if [ -n "$API_PID" ]; then
        kill $API_PID
    fi
    
    # Kill the Next.js process if it exists
    if [ -n "$NEXTJS_PID" ]; then
        kill $NEXTJS_PID
    fi
    
    # Deactivate virtual environment
    deactivate
    
    echo "Servers shut down."
    exit 0
}

# Set up trap to catch termination signals
trap cleanup SIGINT SIGTERM

# Keep script running
echo "Servers are running. Press Ctrl+C to stop."
echo "API is available at http://localhost:8888"
echo "Frontend is available at http://localhost:3000"
echo ""
echo "IMPORTANT: If you experience forecast timeouts:"
echo "1. Try reducing the forecast hours (e.g., use 12 hours instead of 72)"
echo "2. Use the 'Clear Cache' button if you need to regenerate forecasts"
echo "3. The first forecast request after starting the server may take longer"
wait 