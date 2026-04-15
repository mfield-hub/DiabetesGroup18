"""
model_training.py
-----------------
Trains Decision Tree, Random Forest, and XGBoost classifiers.
Saves all models as .pkl files in the models/ directory.
"""

import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add src to path so data_preparation is importable
import sys
sys.path.append(os.path.dirname(__file__))
from data_preparation import get_prepared_data, get_prepared_data_xgb

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def save_model(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"  Saved: {path}")


def train_decision_tree(X_train, X_test, y_train, y_test):
    print("\n--- Decision Tree ---")
    model_path = os.path.join(MODELS_DIR, 'dt_model.pkl')

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("  Loaded existing model.")
    else:
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        save_model(model, 'dt_model.pkl')

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, preds))
    return model


def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n--- Random Forest ---")
    model_path = os.path.join(MODELS_DIR, 'rf_model.pkl')

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("  Loaded existing model.")
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        save_model(model, 'rf_model.pkl')

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, preds))
    return model


def train_xgboost(X_train, X_test, y_train, y_test, le):
    print("\n--- XGBoost ---")
    model_path = os.path.join(MODELS_DIR, 'xgb_model.pkl')

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("  Loaded existing model.")
    else:
        model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', verbosity=0)
        model.fit(X_train, y_train)
        save_model(model, 'xgb_model.pkl')

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc * 100:.2f}%")

    # Decode labels for readable report
    preds_decoded = le.inverse_transform(preds)
    y_test_decoded = le.inverse_transform(y_test)
    print(classification_report(y_test_decoded, preds_decoded))
    return model


def save_feature_columns(X_train):
    """Save the feature column names so the app can align user input."""
    path = os.path.join(MODELS_DIR, 'feature_columns.pkl')
    joblib.dump(list(X_train.columns), path)
    print(f"\nFeature columns saved to {path}")


def train_all():
    print("=== Training Classification Models ===")

    # DT and RF use string labels
    X_train, X_test, y_train, y_test = get_prepared_data()
    dt_model = train_decision_tree(X_train, X_test, y_train, y_test)
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    # XGBoost needs encoded integer labels
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, le = get_prepared_data_xgb()
    xgb_model = train_xgboost(X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, le)

    # Save feature columns (same for all models — same encoding)
    save_feature_columns(X_train)

    print("\n=== All models trained and saved ===")
    return dt_model, rf_model, xgb_model


if __name__ == '__main__':
    train_all()
