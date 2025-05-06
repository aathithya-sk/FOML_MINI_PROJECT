import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('new.csv')
df.head()
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Read the data
df = pd.read_csv('new.csv')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Check columns
print("Columns in dataset:", df.columns.tolist())

# Describe dataset
print(df.describe().T)

# Check for missing values
for col in df.columns:
    temp = df[col].isnull().sum()
    if temp > 0:
        print(f'Column {col} contains {temp} null values.')

# Drop rows with any null values
df = df.dropna()
print("Total rows after removing null values:", len(df))

# Process Dt_Customer column safely
if 'Dt_Customer' in df.columns:
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True, errors='coerce')
    df['day'] = df['Dt_Customer'].dt.day
    df['month'] = df['Dt_Customer'].dt.month
    df['year'] = df['Dt_Customer'].dt.year
    df.drop(['Dt_Customer'], axis=1, inplace=True)
else:
    print("Warning: 'Dt_Customer' column not found. Skipping date processing.")

# Drop unnecessary columns if they exist
for col in ['Z_CostContact', 'Z_Revenue']:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Separate columns by data type
floats, objects = [], []
for col in df.columns:
    if df[col].dtype == object:
        objects.append(col)
    elif df[col].dtype == float:
        floats.append(col)

print("Object columns:", objects)
print("Float columns:", floats)

# Plot categorical distributions
plt.subplots(figsize=(15, 10))
for i, col in enumerate(objects):
    plt.subplot(2, 2, i + 1)
    sb.countplot(data=df, x=col)
plt.tight_layout()
plt.show()

# Marital Status value counts
print(df['Marital_Status'].value_counts())

# Categorical plots with 'Response'
plt.subplots(figsize=(15, 10))
for i, col in enumerate(objects):
    plt.subplot(2, 2, i + 1)
    df_melted = df.melt(id_vars=[col], value_vars=['Response'], var_name='hue')
    sb.countplot(x=col, hue='value', data=df_melted)
plt.tight_layout()
plt.show()

# Encode object columns
for col in objects:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# TSNE visualization
tsne_model = TSNE(n_components=2, random_state=0)
tsne_data = tsne_model.fit_transform(df_scaled)

plt.figure(figsize=(7, 7))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
plt.title("TSNE Projection")
plt.show()

# Finding optimal number of clusters using Elbow Method
error = []
for n_clusters in range(1, 21):
    model = KMeans(init='k-means++', n_clusters=n_clusters, max_iter=500, random_state=22)
    model.fit(df_scaled)
    error.append(model.inertia_)

plt.figure(figsize=(10, 5))
sb.lineplot(x=range(1, 21), y=error, marker="o")
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# KMeans with 5 clusters
model = KMeans(init='k-means++', n_clusters=5, max_iter=500, random_state=22)
segments = model.fit_predict(df_scaled)

# TSNE Scatter plot with cluster segments
df_tsne = pd.DataFrame({
    'x': tsne_data[:, 0],
    'y': tsne_data[:, 1],
    'segment': segments
})

plt.figure(figsize=(7, 7))
sb.scatterplot(x='x', y='y', hue='segment', palette='tab10', data=df_tsne)
plt.title("Customer Segments via KMeans")
plt.show()
