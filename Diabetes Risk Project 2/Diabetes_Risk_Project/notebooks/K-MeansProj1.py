#!/usr/bin/env python
# coding: utf-8

# # Group Patients into Meaningful Lifestyle Based Segments (K-Means)
# Original work by Person 3 (K-Means clustering)

# ## 1) Import Libraries

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

import os
os.getcwd()


# ## 2) Load Dataset

df=pd.read_csv("Diabetes_and_LifeStyle_Dataset_.csv")

#Clean column names
df.columns=df.columns.str.strip()

df.head()


# ## 3) Select Relevant Lifestyle Features

features=['bmi','physical_activity_minutes_per_week','sleep_hours_per_day','alcohol_consumption_per_week','screen_time_hours_per_day']
X=df[features]


# ## 4) Handle Missing Values

X=X.dropna()


# ## 5) Scale Data

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)


# ## 6) Create Elbow Graph to Confirm k=3 is Most Likely

inertia=[]

for k in range(1, 10):
    km=KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# ## 7) Apply K-Means (k=3)

kmeans=KMeans(n_clusters=3, random_state=42)
clusters=kmeans.fit_predict(X_scaled)

df.loc[X.index, 'lifestyle_group']=clusters


# ## 8) Analyze Clusters

cluster_summary=df.groupby('lifestyle_group')[features].mean()
cluster_summary


# ## 9) Interpret Clusters

# ## 10) Visualize Clusters

# ### Pairplot
# NOTE: pairplot is very heavy on 97k rows — sample to 2000 rows if it freezes
# sns.pairplot(df.sample(2000), vars=features, hue='lifestyle_group')
sns.pairplot(df, vars=features, hue='lifestyle_group')
plt.show()


# ### 2D Visualization (PCA)

from sklearn.decomposition import PCA

pca=PCA(n_components=2)
X_pca=pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:, 1], c=clusters)
plt.title("Lifestyle Clusters (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# ### Save Results

df.to_csv("clustered_lifestyle_data.csv", index=False)
