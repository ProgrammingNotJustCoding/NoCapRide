# NoCapRide - Team Saadhana (N-07) - TGBH'25

A modern ride-hailing platform with intelligent demand forecasting and surge balancing.

### [Demo Video](https://youtu.be/bLbfNtwy3UA)

## Project Structure
```
NoCapRide/
├── api.py                  # Flask API for backend services
├── panda.py               # ML-powered forecasting engine
├── requirements.txt       # Python dependencies
├── client/               # Next.js frontend application
│   ├── app/              # App router components
│   │   ├── rides/       # Ride management interface
│   │   └── components/  # Reusable UI components
│   └── package.json     # Frontend dependencies
├── static-data/         # Static data files and resources
├── logs/               # Application logs
└── prev-model-attempt/ # Previous ML model iterations
```

## Features

### For Riders
- Real-time ride booking and tracking
- Transparent surge pricing information
- Scheduled rides
- Saved locations
- Payment method management
- Ride history and statistics

### For Drivers
- Intelligent demand forecasting
- Surge pricing program with priority benefits
- Real-time earnings tracking
- Best times to drive recommendations
- Area-wise demand visualization
- Driver insights dashboard

### Technical Features
- ML-powered demand prediction
- Dynamic surge pricing algorithm
- Real-time data processing
- Interactive data visualizations
- RESTful API architecture
- Modern React/Next.js frontend
- Responsive dark mode UI

## Setup Instructions

### Backend Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Flask API:
   ```bash
   python api.py
   ```

### Frontend Setup
1. Navigate to the client directory:
   ```bash
   cd client
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## API Endpoints

### Demand Forecasting
- `GET /api/forecast` - Get demand forecast for specified region
- `GET /api/nearby-demand` - Get demand in nearby areas
- `GET /api/surge-price` - Calculate surge pricing

### Ride Management
- `POST /api/rides` - Create new ride
- `GET /api/rides` - Get ride history
- `GET /api/rides/:id` - Get specific ride details

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
