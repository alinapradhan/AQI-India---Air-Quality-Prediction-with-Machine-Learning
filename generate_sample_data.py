"""
Generate sample air quality data for India
This creates a synthetic dataset for demonstration purposes
In production, use the actual Kaggle dataset: 
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate dates for 2 years of data
start_date = datetime(2020, 1, 1)
end_date = datetime(2022, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Cities in India
cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 
          'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']

data = []

for date in dates:
    for city in cities:
        # Generate realistic seasonal patterns
        month = date.month
        
        # Winter months (Nov-Feb) tend to have higher pollution
        season_factor = 1.5 if month in [11, 12, 1, 2] else 1.0
        
        # Add some randomness and trends
        pm25 = max(0, np.random.normal(75 * season_factor, 30))
        pm10 = max(0, np.random.normal(120 * season_factor, 40))
        no2 = max(0, np.random.normal(45 * season_factor, 15))
        so2 = max(0, np.random.normal(20 * season_factor, 8))
        co = max(0, np.random.normal(1.2 * season_factor, 0.5))
        o3 = max(0, np.random.normal(60, 20))
        
        # Temperature (Celsius)
        temp = 25 + 10 * np.sin(2 * np.pi * (month - 3) / 12) + np.random.normal(0, 3)
        
        # Humidity (%)
        humidity = 60 + 20 * np.sin(2 * np.pi * (month - 6) / 12) + np.random.normal(0, 10)
        humidity = max(20, min(100, humidity))
        
        # Wind speed (m/s)
        wind_speed = max(0, np.random.normal(3, 1.5))
        
        data.append({
            'Date': date,
            'City': city,
            'PM2.5': round(pm25, 2),
            'PM10': round(pm10, 2),
            'NO2': round(no2, 2),
            'SO2': round(so2, 2),
            'CO': round(co, 2),
            'O3': round(o3, 2),
            'Temperature': round(temp, 2),
            'Humidity': round(humidity, 2),
            'WindSpeed': round(wind_speed, 2)
        })

df = pd.DataFrame(data)

# Calculate AQI based on PM2.5 (simplified US EPA formula)
def calculate_aqi(pm25):
    if pm25 <= 12.0:
        return (50 - 0) / (12.0 - 0.0) * (pm25 - 0.0) + 0
    elif pm25 <= 35.4:
        return (100 - 51) / (35.4 - 12.1) * (pm25 - 12.1) + 51
    elif pm25 <= 55.4:
        return (150 - 101) / (55.4 - 35.5) * (pm25 - 35.5) + 101
    elif pm25 <= 150.4:
        return (200 - 151) / (150.4 - 55.5) * (pm25 - 55.5) + 151
    elif pm25 <= 250.4:
        return (300 - 201) / (250.4 - 150.5) * (pm25 - 150.5) + 201
    else:
        return (500 - 301) / (500.4 - 250.5) * (pm25 - 250.5) + 301

df['AQI'] = df['PM2.5'].apply(calculate_aqi).round(0).astype(int)

# Save to CSV
df.to_csv('data/air_quality_india.csv', index=False)

print(f"Generated {len(df)} records")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Cities: {', '.join(df['City'].unique())}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset statistics:")
print(df.describe())
