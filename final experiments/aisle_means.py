#point the directory back to the parent
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
gmo_final_path = current_file.parent.parent
sys.path.append(str(gmo_final_path))

import matplotlib.pyplot as plt
import numpy as np
from instacart_loader import CSR_Loader
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

loader = CSR_Loader()

aisles, indices = loader.load_reduced_random(filename="hot_customers_aisles", seed=42, n=10000)
aisles_dense = aisles.toarray() if hasattr(aisles, 'toarray') else aisles

col_indices = np.arange(aisles_dense.shape[1])
row_sums = aisles_dense.sum(axis=1)

# Weighted mean: sum(index * count) / sum(count)
row_mean_indices = (aisles_dense * col_indices).sum(axis=1) / (row_sums + 1e-10)  # Mean of each row

mean_val = np.mean(row_mean_indices)
std_val = np.std(row_mean_indices)
print(mean_val, std_val)

plt.figure(figsize=(14, 6))
plt.scatter(range(len(row_mean_indices)), row_mean_indices, alpha=0.5, s=10)
plt.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
plt.axhline(mean_val + std_val, color='orange', linestyle='--', linewidth=1, label=f'Mean ± Std')
plt.axhline(mean_val - std_val, color='orange', linestyle='--', linewidth=1)
plt.xlabel('Customer Index')
plt.ylabel('Mean Aisle Index')
plt.title(f'Customer Shopping Patterns (Mean: {mean_val:.2f}, Std: {std_val:.2f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 137])  # Set to your total number of aisles
plt.tight_layout()
plt.show()


def plot_distributions_scatter(dists, labels=None, figsize=(12, 6), s=50, alpha=0.7):
    """
    Plot multiple distributions as scatter plots.
    
    Parameters:
    -----------
    dists : list of numpy arrays
        List of distributions to plot
    labels : list of str, optional
        Labels for each distribution
    s : float
        Marker size
    alpha : float
        Transparency level (0-1)
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    if labels is None:
        labels = [f'Distribution {i+1}' for i in range(len(dists))]
    
    # Plot each distribution
    for i, dist in enumerate(dists):
        x = np.arange(len(dist))
        plt.scatter(x, dist, s=s, alpha=alpha, label=labels[i])
    
    plt.xlabel('Gene/Feature Index')
    plt.ylabel('Value')
    plt.title('Distributions Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Usage:
# plot_overlapping_distributions(dists)
# plot_overlapping_distributions(dists, labels=['Cluster 0', 'Cluster 1', 'Cluster 2'])

data, indices = loader.load_reduced_random("hot_customers_products", seed=42, n=10000)
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=8)
data = tsvd.fit_transform(data)
print("reduced")

from sklearn.cluster import KMeans
clu = KMeans(n_clusters=20,algorithm="elkan")
clu.fit_predict(data)
chi = calinski_harabasz_score(data, clu.labels_)
print("KMeans wcss", clu.inertia_)
print("CH-Index", chi)

labels = clu.labels_
unique = np.unique(labels)
clusters = [[] for i in range(len(unique))]
for i in range(len(labels)):
    clusters[labels[i]].append(i)

dists = []
for cluster in clusters:
    print("len", len(cluster))
    norm_ent, raw_ent, dist, mean_x = loader.get_entropy(cluster)
    print("aisles", mean_x)
    dists.append(dist)
    norm_ent, raw_ent, dist, mean_x = loader.get_entropy(cluster, "hot_customers_depts")
    print("depts", mean_x)
plot_distributions_scatter(dists)