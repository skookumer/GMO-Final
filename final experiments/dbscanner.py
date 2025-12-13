#point the directory back to the parent
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
gmo_final_path = current_file.parent.parent
sys.path.append(str(gmo_final_path))

from instacart_loader import CSR_Loader
import os
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (                       
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from matplotlib import pyplot as plt
import pandas as pd
from umap import UMAP

l = CSR_Loader()
product_matrix, indices = l.load_reduced_random("hot_customers_products", seed=42, n=10000)
pmnorm = normalize(product_matrix, norm="l1", axis=1)
tsvd = TruncatedSVD(n_components=8)
pmnorm = tsvd.fit_transform(pmnorm)

dbs = DBSCAN(eps=0.005, min_samples=5, metric="cosine")
labels = dbs.fit_predict(pmnorm)
print(labels)

ump = UMAP(n_components=2)
d2d = ump.fit_transform(pmnorm)

# Create the plot
plt.figure(figsize=(12, 8))

# Get unique labels (including -1 for noise)
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

# Create color map
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        # Noise points in black
        color = 'k'
        marker = 'x'
        label_name = 'Noise'
        alpha = 0.3
    else:
        marker = 'o'
        label_name = f'Cluster {label}'
        alpha = 0.6
    
    # Get points for this cluster
    mask = labels == label
    plt.scatter(d2d[mask, 0], d2d[mask, 1], 
                c=[color], label=label_name, 
                marker=marker, s=50, alpha=alpha, edgecolors='none')

plt.xlabel('UMAP 1', fontsize=12)
plt.ylabel('UMAP 2', fontsize=12)
plt.title(f'DBSCAN Clustering (eps=0.5, min_samples=5)\n{n_clusters} clusters, {list(labels).count(-1)} noise points', 
          fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Print summary
print(f"Number of clusters: {n_clusters}")
print(f"Noise points: {list(labels).count(-1)} ({100*list(labels).count(-1)/len(labels):.1f}%)")
print(f"Cluster sizes: {np.bincount(labels[labels >= 0])}")