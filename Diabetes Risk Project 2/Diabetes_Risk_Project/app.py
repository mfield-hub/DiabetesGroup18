"""
app.py
------
Dash web application for Diabetes Risk Prediction and Lifestyle Clustering.
Loads trained .pkl models and provides:
  - Patient data input form
  - Diabetes risk classification (DT / RF / XGBoost)
  - Lifestyle cluster assignment
  - Feature importance chart
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ── Load models ────────────────────────────────────────────────────────────────
dt_model        = joblib.load(os.path.join(MODELS_DIR, 'dt_model.pkl'))
rf_model        = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
xgb_model       = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
kmeans_model    = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
kmeans_scaler   = joblib.load(os.path.join(MODELS_DIR, 'kmeans_scaler.pkl'))
label_encoder   = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))
feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))

CLUSTER_LABELS = {0: 'Active & Healthy', 1: 'Sedentary Risk', 2: 'Moderate Lifestyle'}
CLUSTER_COLORS = {0: '#2ecc71', 1: '#e74c3c', 2: '#f39c12'}

STAGE_COLORS = {
    'No Diabetes':  '#2ecc71',
    'Pre-Diabetes': '#f39c12',
    'Type 1':       '#e67e22',
    'Type 2':       '#e74c3c',
    'Gestational':  '#9b59b6',
}

# ── App ────────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = 'Diabetes Risk Predictor'
server = app.server  # For Render deployment


# ── Helper: build input row for model ─────────────────────────────────────────
def encode_patient(data_dict):
    """Convert form values to a DataFrame aligned with training feature columns."""
    df = pd.DataFrame([data_dict])
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_final = df_encoded.reindex(columns=feature_columns, fill_value=0)
    return df_final


# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    # Header
    dbc.Row([
        dbc.Col(html.H1('🩺 Diabetes Risk Predictor', className='text-center my-4 text-primary'))
    ]),
    dbc.Row([
        dbc.Col(html.P(
            'Enter patient details below to predict diabetes risk stage and lifestyle cluster.',
            className='text-center text-muted mb-4'
        ))
    ]),

    # Input form
    dbc.Card([
        dbc.CardHeader(html.H4('Patient Information', className='mb-0')),
        dbc.CardBody([

            dbc.Row([
                # Column 1 — Demographics
                dbc.Col([
                    html.H5('Demographics', className='text-secondary'),
                    dbc.Label('Age'),
                    dbc.Input(id='age', type='number', value=45, min=1, max=120, className='mb-2'),

                    dbc.Label('Gender'),
                    dcc.Dropdown(id='gender', options=[
                        {'label': 'Male',   'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'},
                        {'label': 'Other',  'value': 'Other'},
                    ], value='Male', clearable=False, className='mb-2'),

                    dbc.Label('Ethnicity'),
                    dcc.Dropdown(id='ethnicity', options=[
                        {'label': 'White',          'value': 'White'},
                        {'label': 'Black',          'value': 'Black'},
                        {'label': 'Asian',          'value': 'Asian'},
                        {'label': 'Hispanic',       'value': 'Hispanic'},
                        {'label': 'Other',          'value': 'Other'},
                    ], value='White', clearable=False, className='mb-2'),

                    dbc.Label('Employment Status'),
                    dcc.Dropdown(id='employment_status', options=[
                        {'label': 'Employed',    'value': 'Employed'},
                        {'label': 'Unemployed',  'value': 'Unemployed'},
                        {'label': 'Retired',     'value': 'Retired'},
                        {'label': 'Student',     'value': 'Student'},
                    ], value='Employed', clearable=False, className='mb-2'),

                    dbc.Label('Smoking Status'),
                    dcc.Dropdown(id='smoking_status', options=[
                        {'label': 'Never',          'value': 'Never'},
                        {'label': 'Former',         'value': 'Former'},
                        {'label': 'Current Smoker', 'value': 'Current'},
                    ], value='Never', clearable=False, className='mb-2'),

                ], md=4),

                # Column 2 — Lifestyle
                dbc.Col([
                    html.H5('Lifestyle', className='text-secondary'),

                    dbc.Label('BMI'),
                    dbc.Input(id='bmi', type='number', value=25.0, step=0.1, className='mb-2'),

                    dbc.Label('Physical Activity (min/week)'),
                    dbc.Input(id='physical_activity', type='number', value=150, className='mb-2'),

                    dbc.Label('Sleep Hours per Day'),
                    dbc.Input(id='sleep_hours', type='number', value=7.0, step=0.5, className='mb-2'),

                    dbc.Label('Alcohol Consumption (drinks/week)'),
                    dbc.Input(id='alcohol', type='number', value=2, className='mb-2'),

                    dbc.Label('Screen Time (hours/day)'),
                    dbc.Input(id='screen_time', type='number', value=4.0, step=0.5, className='mb-2'),

                    dbc.Label('Diet Score (0–10)'),
                    dbc.Input(id='diet_score', type='number', value=5.0, step=0.1, min=0, max=10, className='mb-2'),

                ], md=4),

                # Column 3 — Clinical
                dbc.Col([
                    html.H5('Clinical Measurements', className='text-secondary'),

                    dbc.Label('Systolic BP (mmHg)'),
                    dbc.Input(id='systolic_bp', type='number', value=120, className='mb-2'),

                    dbc.Label('Diastolic BP (mmHg)'),
                    dbc.Input(id='diastolic_bp', type='number', value=80, className='mb-2'),

                    dbc.Label('Fasting Glucose (mg/dL)'),
                    dbc.Input(id='glucose_fasting', type='number', value=95, className='mb-2'),

                    dbc.Label('HbA1c (%)'),
                    dbc.Input(id='hba1c', type='number', value=5.5, step=0.1, className='mb-2'),

                    dbc.Label('BMI (Waist-to-Hip Ratio)'),
                    dbc.Input(id='waist_hip', type='number', value=0.85, step=0.01, className='mb-2'),

                    dbc.Label('Family History of Diabetes'),
                    dcc.Dropdown(id='family_history', options=[
                        {'label': 'Yes', 'value': 1},
                        {'label': 'No',  'value': 0},
                    ], value=0, clearable=False, className='mb-2'),

                    dbc.Label('Hypertension History'),
                    dcc.Dropdown(id='hypertension', options=[
                        {'label': 'Yes', 'value': 1},
                        {'label': 'No',  'value': 0},
                    ], value=0, clearable=False, className='mb-2'),

                ], md=4),
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Label('Select Model'),
                    dcc.Dropdown(id='model_choice', options=[
                        {'label': 'Random Forest (Recommended)', 'value': 'rf'},
                        {'label': 'Decision Tree',               'value': 'dt'},
                        {'label': 'XGBoost',                     'value': 'xgb'},
                    ], value='rf', clearable=False),
                ], md=6),
            ], className='mt-3'),

            dbc.Row([
                dbc.Col([
                    dbc.Button('🔍 Predict', id='predict-btn', color='primary', size='lg',
                               className='mt-4 w-100'),
                ])
            ]),
        ])
    ], className='mb-4 shadow'),

    # Results section
    html.Div(id='results-section'),

], fluid=True)


# ── Callback ───────────────────────────────────────────────────────────────────
@app.callback(
    Output('results-section', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('age', 'value'),
    State('gender', 'value'),
    State('ethnicity', 'value'),
    State('employment_status', 'value'),
    State('smoking_status', 'value'),
    State('bmi', 'value'),
    State('physical_activity', 'value'),
    State('sleep_hours', 'value'),
    State('alcohol', 'value'),
    State('screen_time', 'value'),
    State('diet_score', 'value'),
    State('systolic_bp', 'value'),
    State('diastolic_bp', 'value'),
    State('glucose_fasting', 'value'),
    State('hba1c', 'value'),
    State('waist_hip', 'value'),
    State('family_history', 'value'),
    State('hypertension', 'value'),
    State('model_choice', 'value'),
    prevent_initial_call=True
)
def predict(n_clicks, age, gender, ethnicity, employment, smoking,
            bmi, physical_activity, sleep_hours, alcohol, screen_time,
            diet_score, systolic_bp, diastolic_bp, glucose_fasting,
            hba1c, waist_hip, family_history, hypertension, model_choice):

    # Build raw patient dict (must match training column names)
    patient = {
        'age':                              age,
        'gender':                           gender,
        'ethnicity':                        ethnicity,
        'employment_status':                employment,
        'smoking_status':                   smoking,
        'alcohol_consumption_per_week':     alcohol,
        'physical_activity_minutes_per_week': physical_activity,
        'diet_score':                       diet_score,
        'sleep_hours_per_day':              sleep_hours,
        'screen_time_hours_per_day':        screen_time,
        'family_history_diabetes':          family_history,
        'hypertension_history':             hypertension,
        'cardiovascular_history':           0,
        'bmi':                              bmi,
        'waist_to_hip_ratio':               waist_hip,
        'systolic_bp':                      systolic_bp,
        'diastolic_bp':                     diastolic_bp,
        'heart_rate':                       72,
        'cholesterol_total':                200,
        'hdl_cholesterol':                  55,
        'ldl_cholesterol':                  120,
        'triglycerides':                    150,
        'glucose_fasting':                  glucose_fasting,
        'glucose_postprandial':             140,
        'insulin_level':                    10.0,
        'hba1c':                            hba1c,
        'diabetes_risk_score':              5.0,
        'diagnosed_diabetes':               0,
    }

    # Encode for classification models
    X_input = encode_patient(patient)

    # ── Risk classification ───────────────────────────────────────────────────
    if model_choice == 'dt':
        model = dt_model
        pred_label = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        classes = model.classes_
    elif model_choice == 'xgb':
        model = xgb_model
        pred_idx = model.predict(X_input)[0]
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
        proba = model.predict_proba(X_input)[0]
        classes = label_encoder.classes_
    else:
        model = rf_model
        pred_label = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        classes = model.classes_

    # ── Lifestyle clustering ──────────────────────────────────────────────────
    lifestyle_input = np.array([[
        bmi, physical_activity, sleep_hours, alcohol, screen_time
    ]])
    lifestyle_scaled = kmeans_scaler.transform(lifestyle_input)
    cluster_id = int(kmeans_model.predict(lifestyle_scaled)[0])
    cluster_name = CLUSTER_LABELS.get(cluster_id, f'Cluster {cluster_id}')
    cluster_color = CLUSTER_COLORS.get(cluster_id, '#3498db')

    # ── Probability chart ─────────────────────────────────────────────────────
    proba_fig = go.Figure(go.Bar(
        x=list(classes),
        y=proba,
        marker_color=[STAGE_COLORS.get(c, '#3498db') for c in classes],
        text=[f'{p*100:.1f}%' for p in proba],
        textposition='auto',
    ))
    proba_fig.update_layout(
        title='Prediction Probabilities by Diabetes Stage',
        xaxis_title='Diabetes Stage',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
    )

    # ── Feature importance chart (RF / DT only) ───────────────────────────────
    if model_choice in ('rf', 'dt'):
        importances = model.feature_importances_
        top_n = 10
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_columns[i] for i in indices]
        top_vals = [importances[i] for i in indices]

        importance_fig = go.Figure(go.Bar(
            x=top_vals[::-1],
            y=top_features[::-1],
            orientation='h',
            marker_color='#3498db',
        ))
        importance_fig.update_layout(
            title='Top 10 Feature Importances',
            xaxis_title='Importance',
            template='plotly_white',
        )
        importance_section = dbc.Col([
            dcc.Graph(figure=importance_fig)
        ], md=6)
    else:
        importance_section = dbc.Col([
            dbc.Alert('Feature importance chart available for Decision Tree and Random Forest models.',
                      color='info')
        ], md=6)

    # ── Assemble results ──────────────────────────────────────────────────────
    stage_color = STAGE_COLORS.get(pred_label, '#3498db')

    results = dbc.Card([
        dbc.CardHeader(html.H4('📊 Prediction Results', className='mb-0')),
        dbc.CardBody([

            dbc.Row([
                # Risk result
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5('Predicted Diabetes Stage', className='text-muted'),
                            html.H2(pred_label, style={'color': stage_color, 'fontWeight': 'bold'}),
                            html.P(f'Model used: {model_choice.upper()}', className='text-muted small'),
                        ])
                    ], className='text-center shadow-sm')
                ], md=6),

                # Cluster result
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5('Lifestyle Cluster', className='text-muted'),
                            html.H2(cluster_name, style={'color': cluster_color, 'fontWeight': 'bold'}),
                            html.P(f'Cluster ID: {cluster_id}', className='text-muted small'),
                        ])
                    ], className='text-center shadow-sm')
                ], md=6),
            ], className='mb-4'),

            dbc.Row([
                dbc.Col([dcc.Graph(figure=proba_fig)], md=6),
                importance_section,
            ]),
        ])
    ], className='shadow')

    return results


if __name__ == '__main__':
    app.run(debug=True)
