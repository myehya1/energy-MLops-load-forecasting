import pandas as pd
import numpy as np
import os

# ==============================
# CONFIG
# ==============================

RAW_DATA_PATH = "data/raw/opsd_germany_load.csv"
WEATHER_PATH = "data/raw/weather_germany.csv"
OUTPUT_PATH = "data/processed/final_dataset.csv"

TARGET_COLUMN = "DE_load_actual_entsoe_transparency"


# ==============================
# LOAD ENERGY DATA
# ==============================

def load_energy_data():
    print("Loading energy data...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Convert CET/CEST timestamp to datetime
    df["time"] = pd.to_datetime(df["cet_cest_timestamp"], utc=True)
    df["time"] = df["time"].dt.tz_convert("Europe/Berlin")
    df["time"] = df["time"].dt.tz_localize(None)



    # Select required columns
    columns_needed = [
        "time",
        "DE_load_actual_entsoe_transparency",
        "DE_load_forecast_entsoe_transparency",
        "DE_solar_generation_actual",
        "DE_wind_generation_actual",
        "DE_solar_capacity",
        "DE_wind_capacity",
    ]

    df = df[columns_needed]

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    return df


# ==============================
# LOAD WEATHER DATA
# ==============================

def load_weather_data():
    print("Loading weather data...")
    df = pd.read_csv(WEATHER_PATH)

    df["time"] = pd.to_datetime(df["time"])

    return df


# ==============================
# MERGE DATA
# ==============================

def merge_data(df_energy, df_weather):
    print("Merging datasets...")
    df = pd.merge(df_energy, df_weather, on="time", how="inner")

    df = df.sort_values("time")

    return df


# ==============================
# TIME FEATURES
# ==============================

def create_time_features(df):
    print("Creating time features...")

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


# ==============================
# LAG FEATURES
# ==============================

def create_lag_features(df):
    print("Creating lag features...")

    # 1 hour lag
    df["lag_1"] = df[TARGET_COLUMN].shift(1)

    # 24 hour lag (yesterday same hour)
    df["lag_24"] = df[TARGET_COLUMN].shift(24)

    # 168 hour lag (last week same hour)
    df["lag_168"] = df[TARGET_COLUMN].shift(168)

    return df


# ==============================
# ROLLING FEATURES
# ==============================

def create_rolling_features(df):
    print("Creating rolling features...")

    # 24-hour rolling mean
    df["rolling_mean_24"] = df[TARGET_COLUMN].shift(1).rolling(window=24).mean()

    # 7-day rolling mean
    df["rolling_mean_168"] = df[TARGET_COLUMN].shift(1).rolling(window=168).mean()

    return df


# ==============================
# MAIN PIPELINE
# ==============================

def main():

    os.makedirs("data/processed", exist_ok=True)

    df_energy = load_energy_data()
    df_weather = load_weather_data()

    df = merge_data(df_energy, df_weather)
    df = create_time_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)

    # Drop rows created by lagging
    df = df.dropna()

    df.to_csv(OUTPUT_PATH, index=False)

    print("Feature engineering completed.")
    print("Final dataset shape:", df.shape)


if __name__ == "__main__":
    main()
