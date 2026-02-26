"""
Ship Fuel Consumption — XGBoost Regression Model
=================================================
Run AFTER 01_data_pipeline.py has generated ship_training_data.csv

Requirements:
    pip install pandas numpy scikit-learn xgboost matplotlib shap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DATA_FILE   = "ship_training_data.csv"
MODEL_FILE  = "fuel_model.joblib"

FEATURE_COLS = [
    "sog",
    "stw_knots",
    "wave_height_m",
    "wave_period_s",
    "wind_speed_ms",
    "rel_wind_angle_deg",
    "current_u_ms",
    "current_v_ms",
]

TARGET_COL = "fuel_tph"   # tonnes per hour


# ─────────────────────────────────────────────
# LOAD & PREPARE DATA
# ─────────────────────────────────────────────

def load_data(filepath: str):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    # Drop rows missing features or target
    required = FEATURE_COLS + [TARGET_COL]
    df.dropna(subset=[c for c in required if c in df.columns], inplace=True)
    
    # Remove physically impossible fuel values
    df = df[df[TARGET_COL] > 0]
    df = df[df[TARGET_COL] < df[TARGET_COL].quantile(0.99)]  # remove outliers
    
    print(f"  {len(df):,} rows loaded")
    print(f"  Fuel range: {df[TARGET_COL].min():.4f} — {df[TARGET_COL].max():.4f} t/hr")
    print(f"  Fuel mean:  {df[TARGET_COL].mean():.4f} t/hr")
    
    return df


# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    # Use only features that exist in this dataset
    features = [f for f in FEATURE_COLS if f in df.columns]
    print(f"\nFeatures used: {features}")
    
    X = df[features].values
    y = df[TARGET_COL].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain size: {len(X_train):,} | Test size: {len(X_test):,}")
    
    # ── XGBoost model ──
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="mae",
    )
    
    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    
    return model, features, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate(model, features, X_test, y_test):
    preds = model.predict(X_test)
    
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100
    
    print("\n" + "="*45)
    print("  MODEL EVALUATION")
    print("="*45)
    print(f"  MAE  (tonnes/hr):  {mae:.4f}")
    print(f"  RMSE (tonnes/hr):  {rmse:.4f}")
    print(f"  R²   score:        {r2:.4f}")
    print(f"  MAPE:              {mape:.2f}%")
    print("="*45)
    
    # ── Feature importance ──
    importances = model.feature_importances_
    fi = pd.Series(importances, index=features).sort_values(ascending=False)
    
    print("\n  Feature Importances:")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:<25} {bar} {imp:.3f}")
    
    # ── Plot predicted vs actual ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_test, preds, alpha=0.3, s=10, color="steelblue")
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Fuel (t/hr)")
    axes[0].set_ylabel("Predicted Fuel (t/hr)")
    axes[0].set_title(f"Actual vs Predicted  |  R² = {r2:.3f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Residuals
    residuals = preds - y_test
    axes[1].hist(residuals, bins=50, color="coral", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="black", linewidth=1.5)
    axes[1].set_xlabel("Residual (predicted − actual)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residual Distribution  |  MAE = {mae:.4f}")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150)
    print("\n  Plot saved → model_evaluation.png")
    
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


# ─────────────────────────────────────────────
# INFERENCE EXAMPLE
# ─────────────────────────────────────────────

def predict_fuel(model, features: list, **conditions) -> float:
    """
    Predict fuel burn for given conditions.
    
    Example:
        fuel = predict_fuel(
            model, features,
            sog=12,
            stw_knots=11.5,
            wave_height_m=2.5,
            wave_period_s=8.0,
            wind_speed_ms=10.0,
            rel_wind_angle_deg=30,    # near headwind
            current_u_ms=0.2,
            current_v_ms=-0.1,
        )
        print(f"Estimated fuel: {fuel:.4f} t/hr")
    """
    row = [conditions.get(f, 0) for f in features]
    X = np.array(row).reshape(1, -1)
    return float(model.predict(X)[0])


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    df = load_data(DATA_FILE)
    model, features, X_train, X_test, y_train, y_test = train_model(df)
    metrics = evaluate(model, features, X_test, y_test)
    
    # ── Save model ──
    joblib.dump({"model": model, "features": features}, MODEL_FILE)
    print(f"\nModel saved → {MODEL_FILE}")
    
    # ── Example prediction ──
    print("\n── Example Prediction ──")
    example_fuel = predict_fuel(
        model, features,
        sog=12.0,
        stw_knots=11.2,
        wave_height_m=2.0,
        wave_period_s=8.5,
        wind_speed_ms=9.0,
        rel_wind_angle_deg=45,
        current_u_ms=0.1,
        current_v_ms=0.05,
    )
    print(f"  Ship at 12kn SOG, 2m waves, 9m/s headwind")
    print(f"  → Predicted fuel burn: {example_fuel:.4f} t/hr")
    print(f"  → Daily consumption:   {example_fuel * 24:.2f} tonnes/day")
    
    return model, features, metrics


if __name__ == "__main__":
    main()