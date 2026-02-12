import requests
import pandas as pd
import os

# Create folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 52.52,  # Berlin
    "longitude": 13.41,
    "start_date": "2015-01-01",
    "end_date": "2020-09-30",
    "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
    "timezone": "Europe/Berlin"
}

response = requests.get(url, params=params)
response.raise_for_status()

data = response.json()

# Convert to DataFrame
df_weather = pd.DataFrame(data["hourly"])

# Convert time column properly
df_weather["time"] = pd.to_datetime(df_weather["time"])

df_weather.to_csv("data/raw/weather_germany.csv", index=False)

print("Weather data downloaded successfully.")
