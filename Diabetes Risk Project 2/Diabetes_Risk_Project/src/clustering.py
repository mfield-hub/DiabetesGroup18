"""
clustering.py
-------------
Performs K-Means lifestyle segmentation (k=3) on 5 lifestyle features.
Based on K-MeansProj1.py by Person 3.

Key decisions from original notebook:
  - Features: bmi, physical_activity_minutes_per_week, sleep_hours_per_day,
               alcohol_consumption_per_week, screen_time_hours_per_day
  - k=3 confirmed via elbow method
  - PCA used for 2D visualisation

Saves: kmeans_model.pkl, kmeans_scaler.pkl in models/
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — safe for server/Render
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
sys.path.append(os.path.dirname(__file__))
from data_preparation import load_data, clean_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

LIFESTYLE_FEATURES = [
    'bmi',
    'physical_activity_minutes_per_week',
    'sleep_hours_per_day',
    'alcohol_consumption_per_week',
    'screen_time_hours_per_day'
]

K = 3
CLUSTER_LABELS = {0: 'Active & Healthy', 1: 'Sedentary Risk', 2: 'Moderate Lifestyle'}


def get_lifestyle_data():
    df = load_data()
    df = clean_data(df)
    # Match Person 3's column strip approach
    df.columns = df.columns.str.strip()
    X = df[LIFESTYLE_FEATURES].dropna()
    return df, X


def run_elbow_method(X_scaled, max_k=9, sample_size=10000):
    """
    Elbow method to confirm k=3.
    Samples the data first to avoid slowness on 97k+ rows (noted issue in original notebook).
    """
    print(f"  Running elbow method on {min(sample_size, len(X_scaled))} sampled rows...")
    if len(X_scaled) > sample_size:
        idx = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[idx]
    else:
        X_sample = X_scaled

    inertia = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_sample)
        inertia.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method — Confirming k=3')
    plt.tight_layout()
    os.makedirs(MODELS_DIR, exist_ok=True)
    elbow_path = os.path.join(MODELS_DIR, 'elbow_plot.png')
    plt.savefig(elbow_path)
    plt.close()
    print(f"  Elbow plot saved → {elbow_path}")
    return inertia


def save_pca_plot(X_scaled, clusters):
    """2D PCA scatter plot of clusters — matches Person 3's visualisation."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.4, s=5)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Lifestyle Clusters (PCA 2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    pca_path = os.path.join(MODELS_DIR, 'pca_clusters.png')
    plt.savefig(pca_path, dpi=100)
    plt.close()
    print(f"  PCA cluster plot saved → {pca_path}")


def train_kmeans(run_elbow=True):
    print("\n=== K-Means Lifestyle Clustering ===")

    df, X = get_lifestyle_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if run_elbow:
        run_elbow_method(X_scaled)

    kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'kmeans_scaler.pkl')

    if os.path.exists(kmeans_path):
        kmeans = joblib.load(kmeans_path)
        scaler = joblib.load(scaler_path)
        print("  Loaded existing KMeans model.")
    else:
        print(f"  Training KMeans with k={K}...")
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(kmeans, kmeans_path)
        joblib.dump(scaler, scaler_path)
        print(f"  KMeans model saved → {kmeans_path}")
        print(f"  Scaler saved       → {scaler_path}")

    # Assign clusters back to dataframe (matches Person 3's approach)
    clusters = kmeans.predict(X_scaled)
    df.loc[X.index, 'lifestyle_group'] = clusters

    # Cluster summary (mean of features per cluster)
    print("\n  Cluster Feature Means:")
    cluster_summary = df.groupby('lifestyle_group')[LIFESTYLE_FEATURES].mean()
    print(cluster_summary.to_string())

    # Cluster distribution
    unique, counts = np.unique(clusters, return_counts=True)
    print("\n  Cluster distribution:")
    for cluster, count in zip(unique, counts):
        label = CLUSTER_LABELS.get(int(cluster), f'Cluster {cluster}')
        print(f"    Cluster {int(cluster)} ({label}): {count} patients ({count/len(clusters)*100:.1f}%)")

    # Save PCA visualisation
    save_pca_plot(X_scaled, clusters)

    # Save clustered dataset (as Person 3 did)
    output_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'clustered_lifestyle_data.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n  Clustered data saved → {output_csv}")

    return kmeans, scaler


if __name__ == '__main__':
    train_kmeans()
