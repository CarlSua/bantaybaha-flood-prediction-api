import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib


def train_and_export(data_path: str = 'flood.csv',
                     model_out: str = 'realtime_flood_model.pkl',
                     scaler_out: str = 'realtime_scaler.pkl',
                     config_out: str = 'realtime_config.pkl',
                     random_state: int = 42):
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Fixed thresholds per requirements
    water_level_75th = 5.0
    water_pressure_75th = 30.0
    rain_75th = 60.0

    # Binary indicators
    df['high_water_level'] = (df['water_level_m'] > water_level_75th).astype(int)
    df['high_pressure'] = (df['water_pressure_kpa'] > water_pressure_75th).astype(int)
    df['heavy_rain'] = (df['rain_precip_mm'] > rain_75th).astype(int)

    # Interactions
    df['water_pressure_ratio'] = df['water_level_m'] * df['water_pressure_kpa']
    df['rain_water_interaction'] = df['rain_precip_mm'] * df['water_level_m']

    features = [
        'water_level_m', 'water_pressure_kpa', 'rain_precip_mm',
        'hour', 'day_of_week', 'month', 'is_weekend',
        'high_water_level', 'high_pressure', 'heavy_rain',
        'water_pressure_ratio', 'rain_water_interaction'
    ]
    X = df[features]
    y = df['flood_event']

    # Split
    if y.sum() > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Eval (console)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0.0
    print(f"Accuracy: {acc:.3%}")
    if auc:
        print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save artifacts
    joblib.dump(model, model_out)
    joblib.dump(scaler, scaler_out)
    joblib.dump({
        'features': features,
        'water_level_threshold': water_level_75th,
        'water_pressure_threshold': water_pressure_75th,
        'rain_threshold': rain_75th,
    }, config_out)
    print("Artifacts saved:")
    print(f"- {model_out}\n- {scaler_out}\n- {config_out}")


if __name__ == '__main__':
    train_and_export()

