"""
Deployment script for AQI Prediction Model
This script demonstrates how to use the trained model for predictions
"""

import pandas as pd
from aqi_predictor import AQIPredictor
import sys
import os


def deploy_model():
    """Deploy and use the AQI prediction model"""
    
    print("="*60)
    print("Air Quality Index (AQI) Prediction Model - Deployment")
    print("="*60)
    print()
    
    # Check if model exists
    model_path = 'models/aqi_model.pkl'
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        predictor = AQIPredictor()
        predictor.load_model(model_path)
        print("Model loaded successfully!")
        print()
    else:
        print("No trained model found. Training new model...")
        print()
        
        # Check if data exists
        data_path = 'data/air_quality_india.csv'
        if not os.path.exists(data_path):
            print("Error: Data file not found!")
            print("Please run: python generate_sample_data.py")
            return
        
        # Train model
        predictor = AQIPredictor(model_type='random_forest')
        predictor.train(data_path)
        predictor.save_model(model_path)
        print()
    
    # Load data for predictions
    data_path = 'data/air_quality_india.csv'
    if not os.path.exists(data_path):
        print("Error: Data file not found for predictions!")
        return
    
    df = predictor.load_data(data_path)
    
    # Get list of cities
    cities = sorted(df['City'].unique())
    
    print("="*60)
    print("Making Predictions for Major Indian Cities")
    print("="*60)
    print()
    
    # Make predictions for all cities
    all_predictions = []
    
    for city in cities:
        try:
            predictions = predictor.predict_future(df, city, days_ahead=7)
            all_predictions.append(predictions)
            
            print(f"\n{city}:")
            print("-" * 40)
            for _, row in predictions.iterrows():
                aqi = row['Predicted_AQI']
                date = row['Date'].strftime('%Y-%m-%d')
                
                # Categorize AQI
                if aqi <= 50:
                    category = "Good"
                elif aqi <= 100:
                    category = "Moderate"
                elif aqi <= 150:
                    category = "Unhealthy for Sensitive Groups"
                elif aqi <= 200:
                    category = "Unhealthy"
                elif aqi <= 300:
                    category = "Very Unhealthy"
                else:
                    category = "Hazardous"
                
                print(f"  {date}: AQI {aqi:.0f} ({category})")
        
        except Exception as e:
            print(f"Error predicting for {city}: {e}")
    
    # Combine all predictions
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Save predictions to CSV
        output_path = 'predictions/future_aqi_predictions.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_predictions.to_csv(output_path, index=False)
        print()
        print("="*60)
        print(f"All predictions saved to: {output_path}")
        print("="*60)


def predict_for_city(city_name, days=7):
    """
    Predict AQI for a specific city
    
    Parameters:
    -----------
    city_name : str
        Name of the city
    days : int
        Number of days to predict ahead
    """
    model_path = 'models/aqi_model.pkl'
    
    if not os.path.exists(model_path):
        print("Error: Model not found. Please run deploy_model() first.")
        return
    
    predictor = AQIPredictor()
    predictor.load_model(model_path)
    
    data_path = 'data/air_quality_india.csv'
    df = predictor.load_data(data_path)
    
    predictions = predictor.predict_future(df, city_name, days_ahead=days)
    
    print(f"\nAQI Predictions for {city_name} (Next {days} days):")
    print("="*50)
    print(predictions.to_string(index=False))
    
    return predictions


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Command line usage: python deploy_model.py CityName [days]
        city = sys.argv[1]
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        predict_for_city(city, days)
    else:
        # Deploy for all cities
        deploy_model()
