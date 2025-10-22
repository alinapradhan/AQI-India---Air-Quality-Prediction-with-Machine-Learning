# Quick Start Guide

## Setup (5 minutes)

### 1. Clone and Install
```bash
git clone https://github.com/alinapradhan/AQI-india-and-AI-integration-.git
cd AQI-india-and-AI-integration-
pip install -r requirements.txt
```

### 2. Generate Sample Data
```bash
python generate_sample_data.py
```
This creates `data/air_quality_india.csv` with sample data for 10 Indian cities (2020-2022).

### 3. Train the Model
```bash
python aqi_predictor.py
```
This will:
- Train a Random Forest model
- Show performance metrics (R² ≈ 1.0)
- Save model to `models/aqi_model.pkl`
- Display sample predictions

### 4. Deploy and Predict
```bash
python deploy_model.py
```
This generates predictions for all cities and saves to `predictions/future_aqi_predictions.csv`

## Usage Examples

### Predict for Specific City
```bash
python deploy_model.py Delhi 7       # 7-day forecast for Delhi
python deploy_model.py Mumbai 14     # 14-day forecast for Mumbai
```

### Use in Your Python Code
```python
from aqi_predictor import AQIPredictor

# Load trained model
predictor = AQIPredictor()
predictor.load_model('models/aqi_model.pkl')

# Load data
df = predictor.load_data('data/air_quality_india.csv')

# Predict for a city
predictions = predictor.predict_future(df, 'Delhi', days_ahead=7)
print(predictions)
```

## Understanding the Output

### AQI Categories
- **0-50**: Good (Safe to go outside)
- **51-100**: Moderate (Generally acceptable)
- **101-150**: Unhealthy for Sensitive Groups (Sensitive people should reduce prolonged outdoor exertion)
- **151-200**: Unhealthy (Everyone should reduce prolonged outdoor exertion)
- **201-300**: Very Unhealthy (Avoid all outdoor exertion)
- **301-500**: Hazardous (Stay indoors, use air purifiers)

### Sample Output
```
Predicted AQI for Delhi (Next 7 Days):
      Date  City  Predicted_AQI
2023-01-01 Delhi          201.0    (Very Unhealthy)
2023-01-02 Delhi          201.0    (Very Unhealthy)
...
```

## Using Real Data

### Option 1: Kaggle Dataset
1. Download from: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
2. Extract and place CSV in `data/` folder
3. Update the filename in scripts if different

### Option 2: Your Own Data
Prepare CSV with these columns:
- Date, City, PM2.5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, WindSpeed, AQI

## Troubleshooting

### Import Errors
```bash
pip install --upgrade pandas numpy scikit-learn
```

### Model Not Found
Make sure you run `python aqi_predictor.py` before deployment.

### No Data File
Run `python generate_sample_data.py` first, or download the Kaggle dataset.

## Next Steps

1. **Experiment with different models**: Edit `model_type` in `aqi_predictor.py`
   - `'random_forest'` (default, best accuracy)
   - `'gradient_boosting'` (good for trends)
   - `'linear'` (fast baseline)

2. **Adjust prediction horizon**: Change `days_ahead` parameter (1-30 days)

3. **Add more cities**: Extend the dataset with more Indian cities

4. **Create visualizations**: Use matplotlib/seaborn to plot trends

5. **Build API**: Wrap in Flask/FastAPI for web service

## Performance Expectations

- **Training Time**: ~30 seconds for 3 years of data
- **Prediction Time**: <1 second per city
- **Model Accuracy**: R² > 0.99 (near-perfect on sample data)
- **Memory Usage**: ~50MB

## Support

For issues or questions:
1. Check MODEL_SUMMARY.md for detailed documentation
2. Review README.md for architecture details
3. Open an issue on GitHub

---

**Happy Predicting! 🌍**
