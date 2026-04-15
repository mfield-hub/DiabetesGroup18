"""
data_preparation.py
-------------------
Loads, cleans, encodes, and splits the Diabetes & Lifestyle dataset.
Returns X_train, X_test, y_train, y_test ready for model training.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Diabetes_and_LifeStyle_Dataset_.csv')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

TARGET = 'diabetes_stage'
COLUMNS_TO_DROP = ['income_level', 'education_level']


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def clean_data(df):
    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed {before - after} duplicates. Rows remaining: {after}")

    # Check missing values
    missing = df.isnull().sum()
    if missing.any():
        print("Missing values found:\n", missing[missing > 0])
        df = df.dropna()
    else:
        print("No missing values found.")

    return df


def prepare_features(df, columns_to_drop=COLUMNS_TO_DROP, target=TARGET):
    X = df.drop(columns=[target] + columns_to_drop, errors='ignore')
    y = df[target]
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y


def prepare_features_xgb(df, columns_to_drop=COLUMNS_TO_DROP, target=TARGET):
    """Same as prepare_features but also encodes the target for XGBoost."""
    X_encoded, y = prepare_features(df, columns_to_drop, target)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # Save label encoder for use in the app
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    return X_encoded, y_encoded, le


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_prepared_data():
    """Full pipeline: load → clean → encode → split. Returns all splits."""
    df = load_data()
    df = clean_data(df)
    X_encoded, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X_encoded, y)
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


def get_prepared_data_xgb():
    """Same pipeline but returns encoded target + label encoder for XGBoost."""
    df = load_data()
    df = clean_data(df)
    X_encoded, y_encoded, le = prepare_features_xgb(df)
    X_train, X_test, y_train, y_test = split_data(X_encoded, y_encoded)
    return X_train, X_test, y_train, y_test, le


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_prepared_data()
    print("Data preparation complete.")
    print("Sample classes:", y_train.unique())
