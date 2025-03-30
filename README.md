# NoCapRide - Demand Forecasting & Surge Pricing API
A machine learning-powered ride-hailing platform backend with intelligent demand forecasting and surge pricing capabilities.

## [Demo](https://www.youtube.com/watch?v=bLbfNtwy3UA) --- [Repo Version while recording video](https://github.com/ProgrammingNotJustCoding/NoCapRide/releases/tag/v1.0)

## Project Overview
NoCapRide provides a robust backend API for ride-hailing platforms, focusing on:
- Machine learning-powered demand prediction
- Dynamic surge pricing
- Real-time data processing
- Spatial demand analysis

The system helps both riders and drivers by balancing supply and demand through intelligent forecasting.

## Project Structure
```
NoCapRide/
├── cache/                   # Cached forecasts and API responses (auto-generated)
│   └── forecasts/           # Stored forecast data
├── server/                  # Backend server modules
│   ├── api.py               # API endpoints
│   ├── cache_manager.py     # Cache management logic
│   ├── custom_logger.py     # Custom logging setup
│   ├── data_manager.py      # Data management logic
│   └── forecaster.py        # Forecasting logic
├── archive/                 # Development history
│   ├── attempt-1/           # Initial implementation
│   ├── attempt-2/           # Second iteration with fetch module
│   ├── attempt-3/           # Third iteration
│   ├── data/                # Archive data files
│   └── static-data/         # Static reference data
├── client/                  # Frontend client application
│   ├── app/                 # Next.js app directory
│   ├── components/          # React components
│   ├── lib/                 # Shared utilities
│   ├── public/              # Static assets
│   ├── README.md            # Frontend documentation
│   ├── package.json         # Frontend dependencies
│   ├── next.config.ts       # Next.js configuration
│   └── components.json      # Component configurations
├── logs/                    # Application logs
│   ├── api_log.txt          # API service logs
│   ├── forecast_log.txt     # Forecasting engine logs
│   └── forecast_output.log  # Forecast results
├── ... other default files
```

## Features
### Demand Forecasting
- Time-series forecasting for ride requests by region
- Historical trend analysis
- Feature engineering for temporal patterns
- Random Forest regression model for prediction

### Surge Pricing
- Dynamic pricing based on supply-demand ratio
- Configurable pricing parameters
- Region-specific pricing adjustments
- Real-time price calculation API

### Spatial Analysis
- Nearby high-demand area recommendations
- Region-based demand visualization
- Geographic demand patterns detection

## API Endpoints
### Forecasting
- `GET /api/forecast` - Get demand forecast for specified region and time window
- `GET /api/forecast/all` - Get forecasts for all regions
- `GET /api/regions` - Get available regions for forecasting

### Pricing
- `POST /api/surge_pricing` - Calculate surge pricing for a specific trip
- `POST /api/demand_forecast_ratio` - Get the ratio between forecasted demand and active drivers

### Recommendations
- `GET /api/nearby_high_demand` - Get high-demand locations near a specified region

### Utility
- `GET /api/health` - Health check endpoint

## Setup Instructions
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn api:app --host 127.0.0.1 --port 8888 --reload
   ```

3. Access the API documentation:
   ```
   http://127.0.0.1:8888/docs
   ```

## Configuration
The system is configured to:
- Refresh data from endpoints automatically
- Cache forecast results for performance
- Train/update models on a schedule
- Generate visualizations of forecasts

## Technical Implementation
The system uses:
- **FastAPI**: Modern, high-performance web framework
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning model implementation
- **matplotlib**: Data visualization
- **Threading**: Parallel processing for forecasts

## License
This project is licensed under the MPL v2 License.
