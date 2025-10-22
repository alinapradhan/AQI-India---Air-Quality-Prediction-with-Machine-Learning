"""
Air Quality Index (AQI) Prediction Model for India
This model uses historical air quality data to predict future AQI values
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AQIPredictor:
    """
    Machine Learning model to predict Air Quality Index (AQI)
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the AQI Predictor
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest', 'gradient_boosting', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.city_encoder = LabelEncoder()
        self.feature_names = None
        
    def _create_model(self):
        """Create the specified model"""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            return LinearRegression()
    
    def load_data(self, filepath):
        """
        Load air quality data from CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to the CSV file containing air quality data
            
        Returns:
        --------
        DataFrame with loaded data
        """
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    def create_features(self, df):
        """
        Create features for the model
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe with air quality data
            
        Returns:
        --------
        DataFrame with engineered features
        """
        df = df.copy()
        
        # Temporal features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        df['Quarter'] = df['Date'].dt.quarter
        
        # Cyclical encoding for seasonal patterns
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
        
        # Encode city
        df['City_Encoded'] = self.city_encoder.fit_transform(df['City'])
        
        # Sort by date and city for lag features
        df = df.sort_values(['City', 'Date'])
        
        # Create lag features (previous day's values)
        lag_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity', 'WindSpeed']
        for col in lag_columns:
            if col in df.columns:
                df[f'{col}_lag1'] = df.groupby('City')[col].shift(1)
                df[f'{col}_lag7'] = df.groupby('City')[col].shift(7)  # Previous week
        
        # Rolling averages
        for col in lag_columns:
            if col in df.columns:
                df[f'{col}_rolling7'] = df.groupby('City')[col].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
        
        return df
    
    def prepare_data(self, df, target_column='AQI'):
        """
        Prepare data for training
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe with features
        target_column : str
            Name of the target column to predict
            
        Returns:
        --------
        X : Feature matrix
        y : Target vector
        """
        # Drop rows with NaN values (from lag features)
        df = df.dropna()
        
        # Features to use
        exclude_cols = ['Date', 'City', target_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_column]
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, filepath, target_column='AQI', test_size=0.2):
        """
        Train the model
        
        Parameters:
        -----------
        filepath : str
            Path to the training data CSV file
        target_column : str
            Name of the target column to predict
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        dict with training metrics
        """
        print("Loading data...")
        df = self.load_data(filepath)
        
        print("Creating features...")
        df = self.create_features(df)
        
        print("Preparing data...")
        X, y = self.prepare_data(df, target_column)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {len(self.feature_names)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        print("Evaluating model...")
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        print("\n" + "="*50)
        print("Model Performance Metrics")
        print("="*50)
        print(f"Training RMSE: {metrics['train_rmse']:.2f}")
        print(f"Testing RMSE: {metrics['test_rmse']:.2f}")
        print(f"Training MAE: {metrics['train_mae']:.2f}")
        print(f"Testing MAE: {metrics['test_mae']:.2f}")
        print(f"Training R²: {metrics['train_r2']:.4f}")
        print(f"Testing R²: {metrics['test_r2']:.4f}")
        print("="*50 + "\n")
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
            print()
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : DataFrame or array-like
            Features for prediction
            
        Returns:
        --------
        Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_future(self, df, city, days_ahead=7):
        """
        Predict future AQI values
        
        Parameters:
        -----------
        df : DataFrame
            Historical data
        city : str
            City name to predict for
        days_ahead : int
            Number of days to predict ahead
            
        Returns:
        --------
        DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get the latest data for the city
        city_data = df[df['City'] == city].sort_values('Date')
        
        if len(city_data) == 0:
            raise ValueError(f"No data found for city: {city}")
        
        # Get the last date
        last_date = city_data['Date'].max()
        
        predictions = []
        
        for day in range(1, days_ahead + 1):
            future_date = last_date + timedelta(days=day)
            
            # Create a dummy row with the future date
            future_row = city_data.iloc[-1:].copy()
            future_row['Date'] = future_date
            
            # Append to city data for feature creation
            temp_df = pd.concat([city_data, future_row], ignore_index=True)
            temp_df = self.create_features(temp_df)
            
            # Get the last row (our prediction row)
            pred_row = temp_df.iloc[-1:]
            
            # Prepare features
            feature_cols = [col for col in pred_row.columns if col in self.feature_names]
            X_pred = pred_row[feature_cols]
            
            # Handle missing features
            for col in self.feature_names:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            
            X_pred = X_pred[self.feature_names]
            
            # Make prediction
            pred_aqi = self.predict(X_pred)[0]
            
            predictions.append({
                'Date': future_date,
                'City': city,
                'Predicted_AQI': round(pred_aqi, 2)
            })
        
        return pd.DataFrame(predictions)
    
    def save_model(self, filepath='models/aqi_model.pkl'):
        """
        Save the trained model
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'city_encoder': self.city_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/aqi_model.pkl'):
        """
        Load a trained model
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.city_encoder = model_data['city_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filepath}")


def main():
    """Main function to train and evaluate the model"""
    
    # Create predictor
    predictor = AQIPredictor(model_type='random_forest')
    
    # Train the model
    data_path = 'data/air_quality_india.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run generate_sample_data.py first or download the actual dataset from Kaggle")
        return
    
    metrics = predictor.train(data_path)
    
    # Save the model
    predictor.save_model('models/aqi_model.pkl')
    
    # Make sample predictions
    print("\n" + "="*50)
    print("Sample Future Predictions")
    print("="*50)
    
    df = predictor.load_data(data_path)
    
    # Predict for Delhi for next 7 days
    predictions = predictor.predict_future(df, 'Delhi', days_ahead=7)
    print("\nPredicted AQI for Delhi (Next 7 Days):")
    print(predictions.to_string(index=False))
    
    # Predict for Mumbai
    predictions = predictor.predict_future(df, 'Mumbai', days_ahead=7)
    print("\nPredicted AQI for Mumbai (Next 7 Days):")
    print(predictions.to_string(index=False))


if __name__ == '__main__':
    main()
