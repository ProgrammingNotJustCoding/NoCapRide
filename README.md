# NoCapRide Forecasting System

This project provides a ride request forecasting system with a Flask API backend and a Next.js frontend.

## Components

1. **Forecasting Engine** (`panda.py`): Core forecasting functionality that fetches data, trains models, and generates forecasts.
2. **Flask API** (`api.py`): RESTful API that serves forecast data to the frontend.
3. **Next.js Frontend** (`nextjs-example/`): Web interface for visualizing historical and forecast data.

## Setup

### Prerequisites

- Python 3.8 or higher
- Node.js 14.x or higher (for frontend)
- npm or yarn (for frontend)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NoCapRide.git
cd NoCapRide
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the setup script to create necessary directories:
```bash
mkdir -p logs data/models data/historical data/forecasts data/visualizations
```

### Running the System

You can run both the API and frontend using the provided script:

```bash
./run.sh
```

Or run them separately:

1. Start the Flask API:
```bash
python api.py
```
The API will be available at http://localhost:8888.

2. Start the Next.js frontend:
```bash
cd nextjs-example
npm install
npm run dev
```
The frontend will be available at http://localhost:3000.

## API Endpoints

The Flask API provides the following endpoints:

- `GET /api/health`: Health check endpoint
- `GET /api/forecast`: Get forecast data
- `GET /api/regions`: Get available regions
- `GET /api/historical`: Get historical data
- `GET /api/model/info`: Get model information
- `POST /api/retrain`: Retrain the model
- `POST /api/cache/clear`: Clear the API cache

See the API documentation in `api.py` for more details.

## Frontend

The Next.js frontend provides an interactive dashboard for visualizing historical and forecast data. It includes:

- Time-series charts for historical and forecast data
- Filtering by region and data type
- Adjustable forecast time horizon

## Data Flow

1. The forecasting engine fetches data from external sources
2. The data is processed and used to train a Random Forest model
3. The model generates forecasts for future ride requests
4. The Flask API serves this data to the frontend
5. The Next.js frontend visualizes the data for users

## Directory Structure

```
NoCapRide/
├── api.py                  # Flask API
├── panda.py                # Forecasting engine
├── requirements.txt        # Python dependencies
├── run.sh                  # Script to run both API and frontend
├── data/                   # Data directory
│   ├── models/             # Trained models
│   ├── historical/         # Historical data
│   ├── forecasts/          # Generated forecasts
│   └── visualizations/     # Visualization outputs
├── logs/                   # Log files
└── nextjs-example/         # Next.js frontend
    ├── ForecastComponent.jsx  # Main forecast component
    ├── pages/              # Next.js pages
    └── package.json        # Frontend dependencies
```

## License

See the LICENSE file for details.

## Performance Considerations

### Forecast Generation

Generating forecasts can be computationally intensive, especially for longer time horizons or multiple regions. The system includes several optimizations to improve performance:

1. **Caching**: Forecast results are cached for 30 minutes to avoid redundant calculations
2. **Reduced Estimators**: For faster responses, the number of estimators is temporarily reduced during forecast generation
3. **Timeout Handling**: The frontend includes proper timeout handling and progress indicators

### Handling Timeouts

If you experience forecast timeouts:

1. Try reducing the forecast hours (e.g., use 12 hours instead of 72)
2. Use the "Clear Cache" button if you need to regenerate forecasts
3. The first forecast request after starting the server may take longer (up to 60 seconds)
4. Check the API logs for any errors or performance issues

### Testing the API

You can test the API using the included test script:

```bash
python test_api.py
```

This script will verify that all endpoints are working correctly and will test the caching mechanism. 