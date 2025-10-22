# AQI India - Air Quality Prediction with Machine Learning

A machine learning project to predict future Air Quality Index (AQI) for major cities in India using historical air quality data.

## Dataset

This project uses air quality data for India. The dataset includes pollutant concentrations (PM2.5, PM10, NO2, SO2, CO, O3) along with meteorological data (temperature, humidity, wind speed) for various Indian cities.

**Dataset Source**: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india?resource=download

## Features

- **Data Generation**: Sample data generator for testing and demonstration
- **Machine Learning Model**: Random Forest-based predictor for AQI forecasting
- **Multiple Algorithms**: Support for Random Forest, Gradient Boosting, and Linear Regression
- **Feature Engineering**: Advanced temporal and lag features for better predictions
- **Model Deployment**: Easy-to-use deployment script for making predictions
- **Multi-city Predictions**: Predict AQI for multiple Indian cities simultaneously

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alinapradhan/AQI-india-and-AI-integration-.git
cd AQI-india-and-AI-integration-
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Sample Data

If you don't have the actual Kaggle dataset, generate sample data for testing:

```bash
python generate_sample_data.py
```

This creates a synthetic dataset with realistic air quality patterns for 10 major Indian cities.

### Step 2: Train the Model

Train the prediction model:

```bash
python aqi_predictor.py
```

This will:
- Load the dataset
- Create engineered features
- Train a Random Forest model
- Evaluate performance metrics
- Save the trained model to `models/aqi_model.pkl`
- Display sample predictions

### Step 3: Deploy and Make Predictions

Use the deployment script to make predictions for all cities:

```bash
python deploy_model.py
```

Or predict for a specific city:

```bash
python deploy_model.py Delhi 7
python deploy_model.py Mumbai 14
```

## Model Architecture

### Feature Engineering

The model uses sophisticated feature engineering including:

- **Temporal Features**: Year, month, day, day of week, quarter
- **Cyclical Encoding**: Sine/cosine transformations for seasonal patterns
- **Lag Features**: Previous day and previous week values
- **Rolling Averages**: 7-day moving averages for trend capture
- **Pollutant Data**: PM2.5, PM10, NO2, SO2, CO, O3
- **Meteorological Data**: Temperature, humidity, wind speed

### Model Performance

The Random Forest model achieves:
- High R² score indicating good prediction accuracy
- Low RMSE (Root Mean Square Error)
- Robust performance across different cities

### Supported Cities

- Delhi
- Mumbai
- Bangalore
- Chennai
- Kolkata
- Hyderabad
- Pune
- Ahmedabad
- Jaipur
- Lucknow

## Project Structure

```
AQI-india-and-AI-integration-/
├── data/                          # Data directory
│   └── air_quality_india.csv     # Generated or downloaded dataset
├── models/                        # Saved models directory
│   └── aqi_model.pkl             # Trained model
├── predictions/                   # Predictions output
│   └── future_aqi_predictions.csv
├── generate_sample_data.py        # Sample data generator
├── aqi_predictor.py              # ML model implementation
├── deploy_model.py               # Deployment script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## AQI Categories

The model predicts AQI values which are categorized as:

- **0-50**: Good (Green)
- **51-100**: Moderate (Yellow)
- **101-150**: Unhealthy for Sensitive Groups (Orange)
- **151-200**: Unhealthy (Red)
- **201-300**: Very Unhealthy (Purple)
- **301-500**: Hazardous (Maroon)

## Technical Details

### Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning algorithms
- matplotlib: Data visualization
- seaborn: Statistical visualization
- joblib: Model serialization

### Model Types Available

1. **Random Forest** (default): Ensemble method with high accuracy
2. **Gradient Boosting**: Sequential ensemble method
3. **Linear Regression**: Fast baseline model

## Future Enhancements

- Real-time data integration
- Web dashboard for visualizations
- API endpoint for predictions
- Deep learning models (LSTM, GRU)
- Multi-step ahead forecasting
- Integration with government AQI APIs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms included in the LICENSE file.

## Acknowledgments

- Air quality data from Kaggle
- Indian cities environmental monitoring systems
- Open-source machine learning community
