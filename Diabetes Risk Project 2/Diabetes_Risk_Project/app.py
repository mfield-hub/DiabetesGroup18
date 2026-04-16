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
CLUSTER_COLORS = {0: '#16a34a', 1: '#dc2626', 2: '#d97706'}

STAGE_COLORS = {
    'No Diabetes':  '#16a34a',
    'Pre-Diabetes': '#d97706',
    'Type 1':       '#ea580c',
    'Type 2':       '#dc2626',
    'Gestational':  '#7c3aed',
}

# ── App ────────────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Diabetes Risk Predictor'
server = app.server


# ── Description generator ─────────────────────────────────────────────────────
def generate_description(pred_label, cluster_name, bmi, hba1c, glucose_fasting,
                         physical_activity, diet_score, sleep_hours,
                         smoking, family_history, hypertension):

    # Opening sentence varies by stage
    if pred_label == 'No Diabetes':
        lines = ["Based on the inputs,show very low or no signs of diabetes, which is a good result."]
    elif pred_label == 'Pre-Diabetes':
        lines = ["Early signs of diabetes detected. Blood sugar levels are creeping up, though not into diabatese territory yet, strongly reccomend to get a check up and watch your health"]
    elif pred_label == 'Type 2':
        lines = ["The prediction points toward Type 2 diabetes. This is the most common form and is mostly due to lifestyle, chnages in lifestyle can help here"]
    elif pred_label == 'Type 1':
        lines = ["The prediction points toward Type 1 diabetes. Unlike Type 2, this is autoimmune in nature and isn't caused by lifestyle — insulin therapy is typically needed."]
    elif pred_label == 'Gestational':
        lines = ["Some indicators associated with gestational diabetes are present. This is worth taking seriously given the implications during pregnancy, so close monitoring is important."]
    else:
        lines = ["A risk classification has been made based on the data provided."]

    # Lifestyle cluster
    if cluster_name == 'Active & Healthy':
        lines.append("On the lifestyle side, things look good. The activity levels and general habits are a positive sign.")
    elif cluster_name == 'Sedentary Risk':
        lines.append("The lifestyle concerns. Low activity and sedentary patterns are one of the bigger drivers of metabolic risk.")
    else:
        lines.append("Lifestyle-wise, things are somewhere in the middle. Okay but definetly room for improvenment.")

    # Clinical values — weave naturally into prose
    flags = []
    if hba1c >= 6.5:
        flags.append(f"the HbA1c of {hba1c}% is above the 6.5% diabetic threshold")
    elif hba1c >= 5.7:
        flags.append(f"HbA1c is sitting at {hba1c}%, which is in the pre-diabetic range")
    if glucose_fasting >= 126:
        flags.append(f"fasting glucose of {glucose_fasting} mg/dL is above the 126 mg/dL cutoff")
    elif glucose_fasting >= 100:
        flags.append(f"fasting glucose of {glucose_fasting} mg/dL is slightly elevated")
    if bmi >= 30:
        flags.append(f"BMI of {bmi} puts them in the obese range")
    elif bmi >= 25:
        flags.append(f"BMI of {bmi} is a little above the healthy range")
    if hypertension:
        flags.append("history of hypertension, which adds to the overall risk")
    if family_history:
        flags.append("family history of diabetes, so genetic risk is a factor")
    if smoking == 'Current':
        flags.append("current smoking is making things harder on insulin sensitivity")

    if len(flags) == 1:
        lines.append(f"Worth noting that {flags[0]}.")
    elif len(flags) == 2:
        lines.append(f"A couple of things stand out: {flags[0]}, and {flags[1]}.")
    elif len(flags) >= 3:
        lines.append(f"A few things are worth flagging — {', '.join(flags[:-1])}, and {flags[-1]}.")

    # Advice — conversational, not a bullet list
    advice = []
    if physical_activity < 90:
        advice.append("getting more movement in, even just working up to 150 minutes a week makes a difference")
    if diet_score < 5:
        advice.append("clean up the diet , particularly cutting back on carbs and adding more fibre")
    if sleep_hours < 6:
        advice.append("get more sleep")
    if bmi >= 25:
        advice.append("gradually working toward a healthier weight, even a 5–10% reduction has benefits")

    if pred_label in ('Type 1', 'Type 2', 'Gestational'):
        lines.append("A proper consultation with a doctor should be the first step.")
    elif pred_label == 'Pre-Diabetes':
        lines.append("It's worth booking a follow-up with a GP for a test to get a clearer picture.")

    if advice:
        if len(advice) == 1:
            lines.append(f"Beyond that, the main thing to focus on would be {advice[0]}.")
        else:
            lines.append(f"On the lifestyle front, the biggest wins would come from {', '.join(advice[:-1])}, and {advice[-1]}.")
    elif pred_label == 'No Diabetes':
        lines.append("Keeping up the current habits and staying on top of routine check-ups is the main thing here.")

    return ' '.join(lines)


# ── Helper ─────────────────────────────────────────────────────────────────────
def encode_patient(data_dict):
    df = pd.DataFrame([data_dict])
    df_encoded = pd.get_dummies(df, drop_first=True)
    return df_encoded.reindex(columns=feature_columns, fill_value=0)


def field(label_text, component):
    return html.Div([
        html.Label(label_text, className='field-label'),
        component,
    ], className='field-wrap')


# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # ── Navbar ──
    html.Nav([
        html.Div([
            html.Span('Diabetes Risk', className='brand-light'),
            html.Span(' Predictor', className='brand-bold'),
        ], className='nav-brand'),
        html.Span('MLG 382 · Group 18', className='nav-tag'),
    ], className='app-nav'),

    # ── Page body ──
    html.Div([

        # Page title
        html.Div([
            html.H1([
                'Diabetes ',
                html.Span('Risk Predictor', className='title-accent'),
            ], className='page-title'),
        ], className='page-header'),

        # ── Input card ──
        html.Div([
            html.Div([
                html.Span('Patient Information', className='card-title-text'),
                html.Span('Enter patient details to get a risk classification and lifestyle cluster.', className='card-title-sub'),
            ], className='card-title'),

            html.Div([

                # Demographics
                html.Div([
                    html.P('Demographics', className='section-label'),
                    field('Age', dbc.Input(id='age', type='number', value=45, min=1, max=120)),
                    field('Gender', dcc.Dropdown(id='gender', options=[
                        {'label': 'Male',   'value': 'Male'},
                        {'label': 'Female', 'value': 'Female'},
                        {'label': 'Other',  'value': 'Other'},
                    ], value='Male', clearable=False)),
                    field('Ethnicity', dcc.Dropdown(id='ethnicity', options=[
                        {'label': 'White',    'value': 'White'},
                        {'label': 'Black',    'value': 'Black'},
                        {'label': 'Asian',    'value': 'Asian'},
                        {'label': 'Hispanic', 'value': 'Hispanic'},
                        {'label': 'Other',    'value': 'Other'},
                    ], value='White', clearable=False)),
                    field('Employment Status', dcc.Dropdown(id='employment_status', options=[
                        {'label': 'Employed',   'value': 'Employed'},
                        {'label': 'Unemployed', 'value': 'Unemployed'},
                        {'label': 'Retired',    'value': 'Retired'},
                        {'label': 'Student',    'value': 'Student'},
                    ], value='Employed', clearable=False)),
                    field('Smoking Status', dcc.Dropdown(id='smoking_status', options=[
                        {'label': 'Never',          'value': 'Never'},
                        {'label': 'Former',         'value': 'Former'},
                        {'label': 'Current Smoker', 'value': 'Current'},
                    ], value='Never', clearable=False)),
                ], className='form-col'),

                # Lifestyle
                html.Div([
                    html.P('Lifestyle', className='section-label'),
                    field('BMI', dbc.Input(id='bmi', type='number', value=25.0, step=0.1)),
                    field('Physical Activity (min/week)', dbc.Input(id='physical_activity', type='number', value=150)),
                    field('Sleep Hours per Day', dbc.Input(id='sleep_hours', type='number', value=7.0, step=0.5)),
                    field('Alcohol Consumption (drinks/week)', dbc.Input(id='alcohol', type='number', value=2)),
                    field('Screen Time (hours/day)', dbc.Input(id='screen_time', type='number', value=4.0, step=0.5)),
                    field('Diet Score (0–10)', dbc.Input(id='diet_score', type='number', value=5.0, step=0.1, min=0, max=10)),
                ], className='form-col'),

                # Clinical
                html.Div([
                    html.P('Clinical Measurements', className='section-label'),
                    field('Systolic BP (mmHg)', dbc.Input(id='systolic_bp', type='number', value=120)),
                    field('Diastolic BP (mmHg)', dbc.Input(id='diastolic_bp', type='number', value=80)),
                    field('Fasting Glucose (mg/dL)', dbc.Input(id='glucose_fasting', type='number', value=95)),
                    field('HbA1c (%)', dbc.Input(id='hba1c', type='number', value=5.5, step=0.1)),
                    field('Waist-to-Hip Ratio', dbc.Input(id='waist_hip', type='number', value=0.85, step=0.01)),
                    field('Family History of Diabetes', dcc.Dropdown(id='family_history', options=[
                        {'label': 'Yes', 'value': 1},
                        {'label': 'No',  'value': 0},
                    ], value=0, clearable=False)),
                    field('Hypertension History', dcc.Dropdown(id='hypertension', options=[
                        {'label': 'Yes', 'value': 1},
                        {'label': 'No',  'value': 0},
                    ], value=0, clearable=False)),
                ], className='form-col'),

            ], className='form-grid'),

            html.Div(className='form-divider'),

            html.Div([
                html.Div([
                    html.Label('Model', className='field-label'),
                    dcc.Dropdown(id='model_choice', options=[
                        {'label': 'Random Forest (Recommended)', 'value': 'rf'},
                        {'label': 'Decision Tree',               'value': 'dt'},
                        {'label': 'XGBoost',                     'value': 'xgb'},
                    ], value='rf', clearable=False),
                ], className='model-select'),
                dbc.Button('Run Prediction', id='predict-btn', color='primary', className='predict-btn'),
            ], className='form-footer'),

        ], className='input-card'),

        # Results
        html.Div(id='results-section', className='mt-4'),

    ], className='page-body'),

])


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

    patient = {
        'age':                                age,
        'gender':                             gender,
        'ethnicity':                          ethnicity,
        'employment_status':                  employment,
        'smoking_status':                     smoking,
        'alcohol_consumption_per_week':       alcohol,
        'physical_activity_minutes_per_week': physical_activity,
        'diet_score':                         diet_score,
        'sleep_hours_per_day':                sleep_hours,
        'screen_time_hours_per_day':          screen_time,
        'family_history_diabetes':            family_history,
        'hypertension_history':               hypertension,
        'cardiovascular_history':             0,
        'bmi':                                bmi,
        'waist_to_hip_ratio':                 waist_hip,
        'systolic_bp':                        systolic_bp,
        'diastolic_bp':                       diastolic_bp,
        'heart_rate':                         72,
        'cholesterol_total':                  200,
        'hdl_cholesterol':                    55,
        'ldl_cholesterol':                    120,
        'triglycerides':                      150,
        'glucose_fasting':                    glucose_fasting,
        'glucose_postprandial':               140,
        'insulin_level':                      10.0,
        'hba1c':                              hba1c,
        'diabetes_risk_score':                5.0,
        'diagnosed_diabetes':                 0,
    }

    X_input = encode_patient(patient)

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

    lifestyle_input = np.array([[bmi, physical_activity, sleep_hours, alcohol, screen_time]])
    lifestyle_scaled = kmeans_scaler.transform(lifestyle_input)
    cluster_id = int(kmeans_model.predict(lifestyle_scaled)[0])
    cluster_name  = CLUSTER_LABELS.get(cluster_id, f'Cluster {cluster_id}')
    cluster_color = CLUSTER_COLORS.get(cluster_id, '#2563eb')
    stage_color   = STAGE_COLORS.get(pred_label, '#2563eb')

    description = generate_description(
        pred_label, cluster_name, bmi, hba1c, glucose_fasting,
        physical_activity, diet_score, sleep_hours,
        smoking, family_history, hypertension
    )

    # Probability chart
    proba_fig = go.Figure(go.Bar(
        x=list(classes),
        y=proba,
        marker_color=[STAGE_COLORS.get(c, '#2563eb') for c in classes],
        text=[f'{p*100:.1f}%' for p in proba],
        textposition='auto',
    ))
    proba_fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Diabetes Stage',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        margin=dict(t=40, b=40, l=40, r=20),
        font=dict(family='Inter, sans-serif', size=12),
    )

    # Feature importance chart
    if model_choice in ('rf', 'dt'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_columns[i] for i in indices]
        top_vals     = [importances[i] for i in indices]
        importance_fig = go.Figure(go.Bar(
            x=top_vals[::-1],
            y=top_features[::-1],
            orientation='h',
            marker_color='#2563eb',
        ))
        importance_fig.update_layout(
            title='Top 10 Feature Importances',
            xaxis_title='Importance',
            template='plotly_white',
            margin=dict(t=40, b=40, l=40, r=20),
            font=dict(family='Inter, sans-serif', size=12),
        )
        chart_right = dbc.Col([dcc.Graph(figure=importance_fig)], md=6)
    else:
        chart_right = dbc.Col([
            dbc.Alert('Feature importance is available for Decision Tree and Random Forest models.',
                      color='info', className='mt-2')
        ], md=6)

    results = html.Div([
        html.Div('Results', className='card-title'),

        # Description box
        html.Div([
            html.P('Summary & Recommendations', className='desc-heading'),
            html.P(description, className='desc-body'),
        ], className='desc-box'),

        # Stat cards
        html.Div([
            html.Div([
                html.P('Predicted Stage', className='stat-label'),
                html.Div(pred_label, className='stat-value', style={'color': stage_color}),
                html.Span(f'{model_choice.upper()} model', className='stat-sub'),
            ], className='stat-card', style={'borderLeftColor': stage_color}),

            html.Div([
                html.P('Lifestyle Cluster', className='stat-label'),
                html.Div(cluster_name, className='stat-value', style={'color': cluster_color}),
                html.Span(f'Cluster {cluster_id}', className='stat-sub'),
            ], className='stat-card', style={'borderLeftColor': cluster_color}),
        ], className='stat-row'),

        # Charts
        dbc.Row([
            dbc.Col([dcc.Graph(figure=proba_fig)], md=6),
            chart_right,
        ], className='mt-3'),

    ], className='results-card')

    return results


if __name__ == '__main__':
    app.run(debug=True)
