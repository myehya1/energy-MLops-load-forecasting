import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib
import os
import json
from datetime import datetime

# ==============================
# CONFIG
# ==============================

DATA_PATH = "data/processed/final_dataset.csv"

# Model-specific output directory
MODEL_NAME = "xgboost"
MODEL_OUTPUT_DIR = f"outputs/{MODEL_NAME}"
MODEL_OUTPUT_PATH = f"{MODEL_OUTPUT_DIR}/model.json"
METRICS_OUTPUT_PATH = f"{MODEL_OUTPUT_DIR}/metrics.json"
FEATURE_IMPORTANCE_PATH = f"{MODEL_OUTPUT_DIR}/feature_importance.csv"

TARGET_COL = "target_24h"
TIME_COL = "time"

# Time-based split ratios (no shuffling)
TRAIN_SIZE = 0.7   # 70% for training
VAL_SIZE = 0.1     # 10% for validation (early stopping)
TEST_SIZE = 0.2    # 20% for final evaluation (completely unseen)

# Columns to exclude from features (avoid data leakage)
EXCLUDE_COLS = [
    TIME_COL,
    TARGET_COL,
    "DE_load_actual_entsoe_transparency",  # Actual load - would cause leakage
]

# XGBoost hyperparameters optimized for CPU and time-series
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",  # Fast histogram-based algorithm for CPU
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,  # Use all CPU cores
    "random_state": 42,
    "early_stopping_rounds": 50,
}


# ==============================
# LOAD DATA
# ==============================

def load_data():
    """Load the feature-engineered dataset."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Ensure time column is datetime
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    
    # Sort by time (critical for time-series)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    
    print(f"Dataset loaded. Shape: {df.shape}")
    print(f"Date range: {df[TIME_COL].min()} to {df[TIME_COL].max()}")
    
    return df


# ==============================
# TIME-BASED TRAIN/VAL/TEST SPLIT
# ==============================

def time_based_split(df, train_size=0.7, val_size=0.1, test_size=0.2):
    """
    Split data based on time - NO SHUFFLING.
    Creates 3 splits: train, validation (for early stopping), test (final evaluation).
    
    Args:
        df: Input dataframe sorted by time
        train_size: Proportion for training (default 0.7)
        val_size: Proportion for validation/early stopping (default 0.1)
        test_size: Proportion for final test evaluation (default 0.2)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    print(f"\nPerforming 3-way time-based split (NO SHUFFLING):")
    print(f"  Train: {train_size*100}% | Validation: {val_size*100}% | Test: {test_size*100}%")
    
    n = len(df)
    train_end_idx = int(n * train_size)
    val_end_idx = int(n * (train_size + val_size))
    
    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()
    
    print(f"\nTrain set:      {len(train_df):6d} samples | {train_df[TIME_COL].min()} to {train_df[TIME_COL].max()}")
    print(f"Validation set: {len(val_df):6d} samples | {val_df[TIME_COL].min()} to {val_df[TIME_COL].max()}")
    print(f"Test set:       {len(test_df):6d} samples | {test_df[TIME_COL].min()} to {test_df[TIME_COL].max()}")
    print(f"\n⚠️  Validation set used ONLY for early stopping")
    print(f"⚠️  Test set completely UNTOUCHED until final evaluation")
    
    return train_df, val_df, test_df


# ==============================
# PREPARE FEATURES AND TARGET
# ==============================

def prepare_features_target(df, exclude_cols):
    """
    Separate features and target, excluding specified columns.
    """
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[TARGET_COL]
    
    print(f"\nFeatures: {len(feature_cols)} columns")
    print(f"Target: {TARGET_COL}")
    
    return X, y, feature_cols


# ==============================
# TRAIN MODEL
# ==============================

def train_model(X_train, y_train, X_val, y_val, params):
    """
    Train XGBoost model with early stopping on VALIDATION set.
    Test set is NOT used here - it remains completely unseen.
    """
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    print(f"\nParameters: {json.dumps(params, indent=2)}")
    
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping using VALIDATION SET ONLY
    # Test set is NOT passed here to avoid any data leakage
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )
    
    print(f"\nTraining completed.")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Best validation score: {model.best_score:.4f}")
    
    return model


# ==============================
# EVALUATE MODEL
# ==============================

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Evaluate model on all three splits using RMSE and MAPE.
    This is the FIRST TIME the test set is being used.
    """
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION")
    print("="*50)
    print("\n⚠️  Test set predictions happening NOW for the first time")
    
    # Predictions on all three sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Calculate MAPE
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
    val_mape = mean_absolute_percentage_error(y_val, y_val_pred) * 100
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    
    # Display results
    print("\n" + "-"*50)
    print("RMSE (Root Mean Squared Error) - in MW")
    print("-"*50)
    print(f"Train:      {train_rmse:8.2f} MW")
    print(f"Validation: {val_rmse:8.2f} MW")
    print(f"Test:       {test_rmse:8.2f} MW  ← Final performance metric")
    
    print("\n" + "-"*50)
    print("MAPE (Mean Absolute Percentage Error) - in %")
    print("-"*50)
    print(f"Train:      {train_mape:7.2f}%")
    print(f"Validation: {val_mape:7.2f}%")
    print(f"Test:       {test_mape:7.2f}%  ← Final performance metric")
    
    # Check for overfitting
    print("\n" + "-"*50)
    print("Overfitting Analysis")
    print("-"*50)
    if test_rmse > val_rmse * 1.1:
        print("⚠️  Warning: Test RMSE is significantly higher than validation")
    elif test_rmse < val_rmse * 0.9:
        print("✓ Test performance is better than validation (good!)")
    else:
        print("✓ Test and validation performance are consistent")
    
    metrics = {
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "test_rmse": float(test_rmse),
        "train_mape": float(train_mape),
        "val_mape": float(val_mape),
        "test_mape": float(test_mape),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(y_test),
        "timestamp": datetime.now().isoformat(),
    }
    
    return metrics, y_test_pred


# ==============================
# FEATURE IMPORTANCE
# ==============================

def save_feature_importance(model, feature_cols, output_path):
    """
    Extract and save feature importance.
    """
    print("\nSaving feature importance...")
    
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    })
    
    importance_df = importance_df.sort_values("importance", ascending=False)
    importance_df.to_csv(output_path, index=False)
    
    print(f"Feature importance saved to: {output_path}")
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))


# ==============================
# SAVE MODEL AND METRICS
# ==============================

def save_model_and_metrics(model, metrics, model_path, metrics_path):
    """
    Save trained model and evaluation metrics.
    """
    print(f"\nSaving model to: {model_path}")
    model.save_model(model_path)
    
    print(f"Saving metrics to: {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nModel and metrics saved successfully.")


# ==============================
# MAIN PIPELINE
# ==============================

def main():
    """
    Main training pipeline for 24-hour ahead load forecasting.
    """
    print("="*50)
    print("24-HOUR AHEAD ELECTRICITY LOAD FORECASTING")
    print("XGBoost Training Pipeline")
    print("="*50)
    
    # Create model-specific output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    print(f"\nModel outputs will be saved to: {MODEL_OUTPUT_DIR}/")
    
    # Load data
    df = load_data()
    
    # Check if target column exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset. "
                        f"Available columns: {df.columns.tolist()}")
    
    # Time-based 3-way split
    train_df, val_df, test_df = time_based_split(df, TRAIN_SIZE, VAL_SIZE, TEST_SIZE)
    
    # Prepare features and target for all three sets
    X_train, y_train, feature_cols = prepare_features_target(train_df, EXCLUDE_COLS)
    X_val, y_val, _ = prepare_features_target(val_df, EXCLUDE_COLS)
    X_test, y_test, _ = prepare_features_target(test_df, EXCLUDE_COLS)
    
    # Train model (uses validation set for early stopping, NOT test set)
    model = train_model(X_train, y_train, X_val, y_val, XGBOOST_PARAMS)
    
    # Final evaluation (FIRST TIME test set is used)
    metrics, y_test_pred = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Save feature importance
    save_feature_importance(model, feature_cols, FEATURE_IMPORTANCE_PATH)
    
    # Save model and metrics
    save_model_and_metrics(model, metrics, MODEL_OUTPUT_PATH, METRICS_OUTPUT_PATH)
    
    print("\n" + "="*50)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)


if __name__ == "__main__":
    main()
