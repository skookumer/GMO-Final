from instacart_loader import CSR_Loader
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (                       
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd

# Gap Statistic Implementation for KMeans Clustering Evaluation
def compute_gap_statistic(X, k, wcss, B=5, random_state=42):
    '''This function computes the Gap Statistic for a given clustering result.
    Parameters:
    - X: The original data array (n_samples x n_features).
    - k: Number of clusters.
    - wcss: Within-cluster sum of squares for the original data.
    - B: Number of reference datasets to generate (default is 5).
    - random_state: Seed for reproducibility (default is 42).
    Returns:
    - gap: The computed Gap Statistic value.
    - sk: The standard deviation of the log Wk* values.
    '''
    rng = np.random.default_rng(random_state)
    n_samples, n_features = X.shape
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)

    log_wk = np.log(wcss)
    log_wkbs = np.zeros(B)

    for b in range(B):
        Xb = rng.uniform(xmin, xmax, size=X.shape)
        km_ref = KMeans(
            n_clusters=k,
            n_init=5,
            max_iter=300,
            random_state=random_state + b,
        )
        km_ref.fit(Xb)
        log_wkbs[b] = np.log(km_ref.inertia_)

    gap = log_wkbs.mean() - log_wk
    sk = np.sqrt(1.0 + 1.0 / B) * log_wkbs.std(ddof=1)
    return gap, sk

def get_cluster_dists(labels, indices=None):
    '''Computes cluster aisle distributions and inter-cluster distances.
    Parameters:
    - labels: Cluster labels for each data point.
    - indices: Optional indices of data points to consider.
    Returns:
    - means: Mean values for each cluster.
    - cluster_aisles: Raw aisle distributions for each cluster.
    - normalized_aisles: PCA-reduced normalized aisle distributions.
    - avg_separation: Average L2 distance between cluster profiles.
    - min_separation: Minimum L2 distance between cluster profiles.
    - dist_matrix: Full distance matrix between cluster profiles.
    '''
    pca = PCA(n_components=4)

    if indices is None:
        indices = np.arange(len(labels))
    
    means, cluster_aisles = l.get_aisles(labels, indices)

    normalized_aisles = []
    for i in range(len(cluster_aisles)):
        normalized_aisles.append(cluster_aisles[i] / cluster_aisles[i].sum())

    normalized_aisles = pca.fit_transform(normalized_aisles)

    cluster_matrix = np.vstack(normalized_aisles)
    cluster_profiles = normalize(cluster_matrix, norm='l1', axis=1)
    dist_matrix = euclidean_distances(cluster_profiles, cluster_profiles)
    mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)

    avg_separation = np.mean(dist_matrix[mask])
    min_separation = np.min(dist_matrix[mask])

    return means, cluster_aisles, normalized_aisles, avg_separation, min_separation, dist_matrix

# load data and set up PCA and KMeans instances
l = CSR_Loader()
pc = PCA(n_components=6) #from sindico
km = KMeans(n_clusters=4)

'''REPLICATING SINDICO ON REDUCED DATA. DISCREPANCIES DUE TO TRANSACTION COUNT'''

aisle_matrix, indices = l.load_reduced_random(filename="hot_customers_aisles", seed=42, n=10000)
# aisle_matrix = l.load("hot_customers_aisles")

aisle_matrix = normalize(aisle_matrix, norm="l1", axis=1)
am_reduced = pc.fit_transform(aisle_matrix)[:, [4, 1]]
labels = km.fit_predict(am_reduced)
# Plotting the clusters based on the reduced dimensions (dimensions 1 and 4)
plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    am_reduced[:, 0],  # X-axis data (your first selected dimension)
    am_reduced[:, 1],  # Y-axis data (your second selected dimension)
    c=labels,  # Colors are determined by the cluster label
    cmap='viridis',    # A good default color map for clusters
    s=20,              # Marker size
    alpha=0.6          # Transparency
)

plt.title('K-Means Clustering of Reduced Aisle Data (Dimensions 1 vs 4)')
plt.xlabel('Principal Component 2 (PC2 / Index 1)')
plt.ylabel('Principal Component 5 (PC5 / Index 4)')

# Create a legend to map colors to cluster numbers
legend = plt.legend(*scatter.legend_elements(), 
                    loc="lower left", title="Clusters")
plt.gca().add_artist(legend)

plt.grid(True, linestyle='--', alpha=0.5)
plt.colorbar(scatter, label='Cluster ID')
plt.show()

means, cluster_aisles, norm_aisles, avg_separation, min_separation, dist_matrix = get_cluster_dists(labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    dist_matrix, 
    annot=True, 
    fmt=".3f", 
    cmap="Reds", 
    xticklabels=[f"Cluster {i}" for i in range(len(cluster_aisles))],
    yticklabels=[f"Cluster {i}" for i in range(len(cluster_aisles))]
)
plt.title("L2 Distances Between Cluster Distributions")
plt.show()

print(f"Average Distance between Cluster Centers: {avg_separation:.4f}")
print(f"Minimum Distance (The 'Weakest Link'):    {min_separation:.4f}")



'''SOME OTHER PLOTTING. TOTALLY NOT NECESSARY'''

data = norm_aisles

plt.figure(figsize=(15, 8))

# Loop through each cluster
for i, counts in enumerate(data):
    
    aisle_indices = np.arange(len(counts))
    
    # 1. Plot the outline (Line plot) - stronger alpha to define the shape
    # We switch from scatter to plot to get a nice continuous upper edge
    line = plt.plot(
        aisle_indices, 
        counts, 
        label=f'Cluster {i}', 
        linewidth=2,
        alpha=0.8
    )
    
    # 2. Shade the area under the curve
    # We use the color of the line we just plotted (line[0].get_color())
    plt.fill_between(
        aisle_indices, 
        counts, 
        color=line[0].get_color(), # Matches the line color
        alpha=0.2                  # Low opacity (20%) to see overlaps clearly
    )

# Formatting
plt.title('Distribution of Purchases by Aisle per Cluster (Area View)')
plt.xlabel('Aisle ID (0-133)')
plt.ylabel('Total Items Purchased') # or 'Proportion' if normalized
plt.legend(title="Clusters")
plt.grid(True, linestyle='--', alpha=0.4)

plt.xticks(np.arange(0, len(data), 3))
plt.xlim(0, len(data))             # Ensure the shading starts/ends cleanly
plt.ylim(bottom=0)           # Start Y-axis at 0

plt.show()

# TSVD + KMeans sweep
TSVD_COMPONENTS = [4, 8, 16]
K_VALUES = [4, 6, 8, 10]

results = [] # to store results

# TSVD + KMeans sweep and metrics calculation
for n_comp in TSVD_COMPONENTS:
    tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_svd = tsvd.fit_transform(aisle_matrix)
    for k in K_VALUES:
        km_z = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=42)
        labels_z = km_z.fit_predict(X_svd)
        wcss_z = km_z.inertia_
        sil_z = silhouette_score(X_svd, labels_z)
        ch_z = calinski_harabasz_score(X_svd, labels_z)
        db_z = davies_bouldin_score(X_svd, labels_z)
        _, _, _, avg_sep_z, min_sep_z, _ = get_cluster_dists(labels_z, indices)
        gap_z, gap_se_z = compute_gap_statistic(X_svd, k, wcss_z)
        print(
            f"UPDATED-TSVD: d={n_comp:2d}, k={k:2d} | "
            f"WCSS={wcss_z:12.1f} | Sil={sil_z:6.3f} | "
            f"CH={ch_z:10.1f} | DB={db_z:6.3f} | "
            f"avg_sep={avg_sep_z:6.3f} | min_sep={min_sep_z:6.3f}"
        )

        results.append({
            "svd_dim": n_comp,
            "k": k,
            "wcss": wcss_z,
            "silhouette": sil_z,
            "ch_index": ch_z,
            "db_index": db_z,
            "gap": gap_z, 
            "gap_se": gap_se_z,          
            "avg_separation": avg_sep_z,
            "min_separation": min_sep_z,
        })

# converts metrics to DataFrame and exports to CSV
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("tsvd_kmeans_metrics.csv", index=False)
print("Exported TSVD + KMeans metrics to tsvd_kmeans_metrics.csv")

# line plots for WCSS, Silhouette, CH, DB vs k (one line per TSVD dim)
metric_info = [
    ("wcss", "WCSS (Within-Cluster Sum of Squares)"),
    ("silhouette", "Silhouette Score"),
    ("ch_index", "Calinski–Harabasz Index"),
    ("db_index", "Davies–Bouldin Index"),
    ("gap", "Gap Statistic"),              
]

for metric, ylabel in metric_info:
    plt.figure(figsize=(8, 6))
    for n_comp in TSVD_COMPONENTS:
        subset = metrics_df[metrics_df["svd_dim"] == n_comp].sort_values("k")
        plt.plot(subset["k"], subset[metric], marker="o", label=f"d={n_comp}")
    plt.xlabel("Number of clusters k")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs k for different TSVD dimensions")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# line plots for average and minimum aisle separation vs k
sep_info = [
    ("avg_separation", "Average L2 Separation Between Clusters"),
    ("min_separation", "Minimum L2 Separation Between Clusters"),
]

for metric, ylabel in sep_info:
    plt.figure(figsize=(8, 6))
    for n_comp in TSVD_COMPONENTS:
        subset = metrics_df[metrics_df["svd_dim"] == n_comp].sort_values("k")
        plt.plot(subset["k"], subset[metric], marker="o", label=f"d={n_comp}")
    plt.xlabel("Number of clusters k")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs k for different TSVD dimensions")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

# heatmap for Silhouette scores across (svd_dim, k)
pivot_sil = metrics_df.pivot(index="svd_dim", columns="k", values="silhouette")
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_sil, annot=True, fmt=".3f", cmap="viridis")
plt.title("Silhouette Score Heatmap (TSVD dimension vs k)")
plt.xlabel("Number of clusters k")
plt.ylabel("TSVD dimension")
plt.tight_layout()
plt.show()