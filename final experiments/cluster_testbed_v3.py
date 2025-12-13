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
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
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

    cluster_matrix = np.vstack(normalized_aisles)
    cluster_profiles = normalize(cluster_matrix, norm='l1', axis=1)
    normalized_aisles = pca.fit_transform(cluster_profiles)
    dist_matrix = euclidean_distances(cluster_profiles, cluster_profiles)
    mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)

    avg_separation = np.mean(dist_matrix[mask])
    min_separation = np.min(dist_matrix[mask])

    return means, cluster_aisles, normalized_aisles, avg_separation, min_separation, dist_matrix

# load data and set up PCA and KMeans instances
l = CSR_Loader()
# pc = PCA(n_components=6) #from sindico
# km = KMeans(n_clusters=4)

# '''REPLICATING SINDICO ON REDUCED DATA. DISCREPANCIES DUE TO TRANSACTION COUNT'''

# aisle_matrix, a_indices = l.load_reduced_random(filename="hot_customers_aisles", seed=42, n=10000)
# # aisle_matrix = l.load("hot_customers_aisles")

# aisle_matrix = normalize(aisle_matrix, norm="l1", axis=1)
# am_reduced = pc.fit_transform(aisle_matrix)[:, [4, 1]]
# labels = km.fit_predict(am_reduced)
# # Plotting the clusters based on the reduced dimensions (dimensions 1 and 4)
# plt.figure(figsize=(10, 8))

# scatter = plt.scatter(
#     am_reduced[:, 0],  # X-axis data (your first selected dimension)
#     am_reduced[:, 1],  # Y-axis data (your second selected dimension)
#     c=labels,  # Colors are determined by the cluster label
#     cmap='viridis',    # A good default color map for clusters
#     s=20,              # Marker size
#     alpha=0.6          # Transparency
# )

# plt.title('K-Means Clustering of Reduced Aisle Data (Dimensions 1 vs 4)')
# plt.xlabel('Principal Component 2 (PC2 / Index 1)')
# plt.ylabel('Principal Component 5 (PC5 / Index 4)')

# # Create a legend to map colors to cluster numbers
# legend = plt.legend(*scatter.legend_elements(), 
#                     loc="lower left", title="Clusters")
# plt.gca().add_artist(legend)

# plt.grid(True, linestyle='--', alpha=0.5)
# plt.colorbar(scatter, label='Cluster ID')
# # plt.show()

# means, cluster_aisles, norm_aisles, avg_separation, min_separation, dist_matrix = get_cluster_dists(labels, indices=a_indices)

# fig, ax = plt.subplots(figsize=(8, 6))

# sns.heatmap(dist_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
#             square=True, linewidths=0.5, cbar_kws={'label': 'Distance'},
#             xticklabels=[f'Cluster {i}' for i in range(len(dist_matrix))],
#             yticklabels=[f'Cluster {i}' for i in range(len(dist_matrix))],
#             ax=ax)

# ax.set_title('Inter-Cluster Distance Matrix', fontsize=14, fontweight='bold')
# plt.tight_layout()
# plt.show()

# cluster_aisles = [np.sqrt(aisle / sum(aisle)) for aisle in cluster_aisles]

# n_aisles = len(cluster_aisles[0])
# aisle_indices = np.arange(n_aisles)
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# fig, ax = plt.subplots(figsize=(16, 6))

# x = np.arange(n_aisles)

# for i, arr in enumerate(cluster_aisles):
#     ax.fill_between(x, 0, arr, color=colors[i], alpha=0.5, label=f'Cluster {i}')

# ax.set_xlabel('Aisle Index', fontsize=12)
# ax.set_ylabel('Sqrt(norm count)', fontsize=12)
# ax.set_title('Aisle Distributions Across 4 Clusters', fontsize=14, fontweight='bold')
# ax.legend()
# ax.grid(axis='y', alpha=0.3)

# plt.tight_layout()
# plt.show()

# data = norm_aisles
# plt.figure(figsize=(15, 8))

# # Loop through each cluster
# for i, counts in enumerate(data):
    
#     aisle_indices = np.arange(len(counts))
    
#     # 1. Plot the outline (Line plot) - stronger alpha to define the shape
#     # We switch from scatter to plot to get a nice continuous upper edge
#     line = plt.plot(
#         aisle_indices, 
#         counts, 
#         label=f'Cluster {i}', 
#         linewidth=2,
#         alpha=0.8
#     )
    
#     # 2. Shade the area under the curve
#     # We use the color of the line we just plotted (line[0].get_color())
#     plt.fill_between(
#         aisle_indices, 
#         counts, 
#         color=line[0].get_color(), # Matches the line color
#         alpha=0.2                  # Low opacity (20%) to see overlaps clearly
#     )

# # Formatting
# plt.title('Distribution of Purchases by Aisle per Cluster (Area View)')
# plt.xlabel('Aisle ID (0-133)')
# plt.ylabel('Total Items Purchased') # or 'Proportion' if normalized
# plt.legend(title="Clusters")
# plt.grid(True, linestyle='--', alpha=0.4)

# plt.xticks(np.arange(0, len(data), 3))
# plt.xlim(0, len(data))             # Ensure the shading starts/ends cleanly
# plt.ylim(bottom=0)           # Start Y-axis at 0

# plt.show()





if os.path.exists(Path(__file__).parent / "KMeans_clusters.parquet") is False:
    TSVD_COMPONENTS = [10]
    K_VALUES = [5]

    product_matrix, p_indices = l.load_reduced_random(filename="hot_customers_products", seed=42, n=10000)
    aisle_matrix, p_indices = l.load_reduced_random(filename="hot_customers_aisles", seed=42, n=10000)

    product_matrix_normed = normalize(product_matrix, norm="l1", axis=1)
    aisle_matrix_normed = normalize(aisle_matrix, norm="l1", axis=1)

    kmeans_results = []
    for n_comp in TSVD_COMPONENTS:
        tsvd = TruncatedSVD(n_components=n_comp)
        X_svd = tsvd.fit_transform(product_matrix_normed)
        for k in K_VALUES:
            km_z = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=42)
            labels_z = km_z.fit_predict(X_svd)
            kmeans_results.append({"model": "KMeans_products", "tsvd": n_comp, "k": k, "labels": labels_z, "WCSS": km_z.inertia_})

    kmeans_aisles = []
    for n_comp in TSVD_COMPONENTS:
        tsvd = TruncatedSVD(n_components=n_comp)
        X_svd = tsvd.fit_transform(aisle_matrix_normed)
        for k in K_VALUES:
            km_z = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=42)
            labels_z = km_z.fit_predict(X_svd)
            kmeans_aisles.append({"model": "KMeans_aisles", "tsvd": n_comp, "k": k, "labels": labels_z, "WCSS": km_z.inertia_})

    sindico_results = []
    # for n_comp in TSVD_COMPONENTS:
    #     pca = PCA(n_components=6)
    #     X_pca = pca.fit_transform(aisle_matrix_normed)[:, [1, 4]]
    #     for k in K_VALUES:
    #         km_s = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=42)
    #         labels_s = km_s.fit_predict(X_pca)
    #         sindico_results.append({"model": "KMeans_sindico", "tsvd": 6, "k": k, "labels": labels_s, "WCSS": km_s.inertia_})

    dampah = [0.7, 0.9]
    affinity_results = []
    # for n_comp in TSVD_COMPONENTS:
    #     tsvd = TruncatedSVD(n_components=n_comp)
    #     X_svd = tsvd.fit_transform(aisle_matrix_normed)
    #     for damp in dampah:
    #         ap = AffinityPropagation(damping=damp)
    #         labels = ap.fit_predict(X_svd)
    #         centers = ap.cluster_centers_

    #         le = LabelEncoder()
    #         labels = le.fit_transform(labels)

    #         # Calculate WCSS manually
    #         wcss = 0
    #         for i in range(len(centers)):
    #             cluster_points = X_svd[labels == i]
    #             distances = pairwise_distances(cluster_points, [centers[i]], squared=True)
    #             wcss += np.sum(distances)

    #         affinity_results.append({"model": f"Affinity Propagation, {damp}", "tsvd": n_comp, "k": len(np.unique(labels)), "labels": labels, "WCSS": wcss})
        
    
    daria_results = []
    for n_comp in TSVD_COMPONENTS:
        pca = PCA(n_components=10)
        X_pca = pca.fit_transform(aisle_matrix_normed)
        for k in K_VALUES:
            km_s = KMeans(n_clusters=k, n_init=20, max_iter=300, random_state=42)
            labels_s = km_s.fit_predict(X_pca)
            daria_results.append({"model": "KMeans_daria", "tsvd": 10, "k": k, "labels": labels_s, "WCSS": km_s.inertia_})

    results = kmeans_results + kmeans_aisles + sindico_results + daria_results + affinity_results
    df = pd.DataFrame(results)
    df.to_parquet(Path(__file__).parent / "KMeans_clusters.parquet")

km_results = pd.read_parquet(Path(__file__).parent / "KMeans_clusters.parquet")
print(km_results.head())

parquet_dir = gmo_final_path / "parquet_files"
nsga_files = [f for f in parquet_dir.iterdir() if 'NSGA' in f.name]

dfs = []
for file in nsga_files:
    df = pd.read_parquet(file)
    df['model'] = file.stem + '_' + df['type'].astype(str)
    df['labels'] = df['labels'].apply(lambda x: LabelEncoder().fit_transform(x))
    df = df.nsmallest(1, 'WCSS')

    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# print(combined_df.iloc[10]["labels"])


final_df = pd.concat([km_results, combined_df], ignore_index=True)
final_df.loc[final_df['model'].str.contains('Affinity'), 'k'] = 10


data = final_df



algorithms = data['model'].unique()
matrix, indices = l.load_reduced_random("hot_customers_aisles", seed=42, n=10000)
data_matrix_norm = normalize(matrix, norm="l1", axis=1)

results = []
for idx, row in data.iterrows():
    print(idx)
    model = row["model"]
    dim = row["tsvd"]
    clusters = row["k"]
    labels = np.array(row["labels"])
    wcss = row["WCSS"]

    result = {"model": model, "dim": dim, "clusters": clusters, "WCSS": wcss}

    if "daria" in model:
        matrix, indices = l.load_reduced_random("hot_customers_aisles", seed=42, n=10000)
        data_matrix_norm = normalize(matrix, norm="l1", axis=1)
        pca = PCA(n_components = dim)
        dmatrix = pca.fit_transform(data_matrix_norm)
    
    if "sindico" in model:
        matrix, indices = l.load_reduced_random("hot_customers_aisles", seed=42, n=10000)
        data_matrix_norm = normalize(matrix, norm="l1", axis=1)
        pca = PCA(n_components = dim)
        dmatrix = pca.fit_transform(data_matrix_norm)
        dmatrix = dmatrix[:, [1, 4]]

    elif "aisles" in model:
        matrix, indices = l.load_reduced_random("hot_customers_aisles", seed=42, n=10000)
        data_matrix_norm = normalize(matrix, norm="l1", axis=1)
        tsvd = TruncatedSVD(n_components = dim)
        dmatrix = tsvd.fit_transform(data_matrix_norm)
    else:
        matrix, indices = l.load_reduced_random("hot_customers_products", seed=42, n=10000)
        data_matrix_norm = normalize(matrix, norm="l1", axis=1)
        tsvd = TruncatedSVD(n_components = dim)
        dmatrix = tsvd.fit_transform(data_matrix_norm)

    result["silhouette"] = silhouette_score(dmatrix, labels)
    result["ch_index"] = calinski_harabasz_score(dmatrix, labels)
    result["db_index"] = davies_bouldin_score(dmatrix, labels)
    _, _, _, avg_sep_z, _, _ = get_cluster_dists(labels, indices)
    result["aisle_sep"] = avg_sep_z
    gap_z, gap_se_z = compute_gap_statistic(dmatrix, clusters, wcss)
    result["gap"] = gap_z
    result["gap_se"] = gap_se_z 
    results.append(result)

metrics_df = pd.DataFrame(results)

algo_styles = {
    "KMeans_products": {
        "color_cool": "#2E86AB",  # Cool blue
        "color_warm": "#A23B72",  # Warm magenta
        "marker": "o", 
        "linestyle": "-", 
        "label": "KMeans Products"
    },
    "KMeans_aisles": {
        "color_cool": "#06A77D",  # Cool teal
        "color_warm": "#D97D23",  # Warm orange
        "marker": "s", 
        "linestyle": "--", 
        "label": "KMeans Aisles"
    },
    "KMeans_sindico": {
        "color_cool": "#6A4C93",  # Cool purple
        "color_warm": "#C73E1D",  # Warm red
        "marker": "^", 
        "linestyle": "-.", 
        "label": "KMeans Sindico"
    },
    "KMeans_daria": {
        "color_cool": "#2D6A4F",  # Cool forest green
        "color_warm": "#E63946",  # Warm crimson
        "marker": "v", 
        "linestyle": "-", 
        "label": "KMeans Daria"
    },
    "Affinity": {
        "color_cool": "#3D5A80",  # Cool navy
        "color_warm": "#EE964B",  # Warm gold
        "marker": "P", 
        "linestyle": "-", 
        "label": "Affinity Propagation"
    }
}

# NSGA variants with different markers but similar colors
nsga_variants = {
    "markers": ["D", "X", "*", "p", "h"],
    "linestyles": [":", "-.", "--", "-", ":"],
    "color_cool": "#1D7874",  # Cool cyan
    "color_warm": "#F95738",  # Warm coral
}

# Helper function to get style based on algorithm name
def get_style(algorithm):
    if "NSGA" in algorithm:
        # Get all NSGA algorithms and find the index
        nsga_algos = sorted([a for a in algorithms if "NSGA" in a])
        idx = nsga_algos.index(algorithm) if algorithm in nsga_algos else 0
        # Create small offset based on index (-0.2, -0.1, 0, 0.1, 0.2, etc.)
        offset = (idx - len(nsga_algos) // 2) * 0.3
        return {
            "color_cool": nsga_variants["color_cool"],
            "color_warm": nsga_variants["color_warm"],
            "marker": nsga_variants["markers"][idx % len(nsga_variants["markers"])],
            "linestyle": nsga_variants["linestyles"][idx % len(nsga_variants["linestyles"])],
            "label": algorithm,
            "offset": offset
        }
    elif "Affinity" in algorithm:
        style = algo_styles["Affinity"].copy()
        style["offset"] = 0
        return style
    else:
        style = algo_styles.get(algorithm, algo_styles["KMeans_products"]).copy()
        style["offset"] = 0
        return style


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.precision', 4)

# Sort by dimension, model, and clusters for easy reading
all_data = metrics_df.sort_values(['dim', 'model', 'clusters'])

print(f"\n{'='*140}")
print(f"Complete Clustering Metrics Summary")
print(f"{'='*140}\n")
print(all_data.to_string(index=False))
print(f"\n{'='*140}\n")


# dim_data = metrics_df[metrics_df['dim'] == dim]

# # Create figure with subplots - increased figure size and better spacing
# fig, axes = plt.subplots(2, 2, figsize=(20, 16))
# fig.suptitle(f'Clustering Metrics Comparison (Dimension = {dim})', 
#                 fontsize=18, fontweight='bold', y=0.995)

# # Plot 1: Silhouette Score (COOL) and WCSS (WARM)
# ax1 = axes[0, 0]
# ax1_twin = ax1.twinx()

# for algorithm in algorithms:
#     algo_data = dim_data[dim_data['model'] == algorithm].sort_values('clusters')
#     style = get_style(algorithm)
    
#     # Silhouette in COOL colors
#     ax1.plot(algo_data['clusters'], algo_data['silhouette'], 
#             marker=style['marker'], color=style['color_cool'], linestyle=style['linestyle'],
#             label=f'{algorithm} - Silhouette', linewidth=2.5, markersize=8)
    
#     # WCSS in WARM colors
#     ax1_twin.plot(algo_data['clusters'], algo_data['WCSS'],
#                     marker=style['marker'], color=style['color_warm'], linestyle=style['linestyle'],
#                     label=f'{algorithm} - WCSS', linewidth=2.5, markersize=7)

# ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
# ax1.set_ylabel('Silhouette Score', fontsize=12, color='#2E5266')
# ax1.tick_params(axis='y', labelcolor='#2E5266')
# ax1_twin.set_ylabel('WCSS', fontsize=12, color='#9B2915')
# ax1_twin.tick_params(axis='y', labelcolor='#9B2915')
# ax1.set_title('Silhouette Score (Cool) & WCSS (Warm)', fontsize=13, fontweight='bold', pad=15)
# ax1.grid(True, alpha=0.3)
# ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
# ax1_twin.legend(loc='upper right', fontsize=8, framealpha=0.9)

# # Plot 2: Calinski-Harabasz Index (single axis - use cool colors)
# ax2 = axes[0, 1]

# for algorithm in algorithms:
#     algo_data = dim_data[dim_data['model'] == algorithm].sort_values('clusters')
#     style = get_style(algorithm)
    
#     ax2.plot(algo_data['clusters'], algo_data['ch_index'], 
#             marker=style['marker'], color=style['color_cool'], linestyle=style['linestyle'],
#             label=algorithm, linewidth=2.5, markersize=8)

# ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
# ax2.set_ylabel('Calinski-Harabasz Index', fontsize=12)
# ax2.set_title('Calinski-Harabasz Index (Higher is Better)', fontsize=13, fontweight='bold', pad=15)
# ax2.legend(fontsize=9, framealpha=0.9, loc='best')
# ax2.grid(True, alpha=0.3)

# # Plot 3: Davies-Bouldin Index (COOL) & Gap Statistic (WARM)
# ax3 = axes[1, 0]
# ax3_twin = ax3.twinx()

# for algorithm in algorithms:
#     algo_data = dim_data[dim_data['model'] == algorithm].sort_values('clusters')
#     style = get_style(algorithm)
    
#     # DB-Index in COOL colors
#     ax3.plot(algo_data['clusters'], algo_data['db_index'], 
#             marker=style['marker'], color=style['color_cool'], linestyle=style['linestyle'],
#             label=f'{algorithm} - DB-Index', linewidth=2.5, markersize=8)
    
#     # Gap in WARM colors
#     ax3_twin.errorbar(algo_data['clusters'], algo_data['gap'],
#                         yerr=algo_data['gap_se'],
#                         marker=style['marker'], color=style['color_warm'], linestyle=style['linestyle'],
#                         label=f'{algorithm} - Gap', linewidth=2.5,
#                         capsize=5, markersize=7)

# ax3.set_xlabel('Number of Clusters (k)', fontsize=12)
# ax3.set_ylabel('Davies-Bouldin Index', fontsize=12, color='#2E5266')
# ax3.tick_params(axis='y', labelcolor='#2E5266')
# ax3_twin.set_ylabel('Gap Statistic', fontsize=12, color='#9B2915')
# ax3_twin.tick_params(axis='y', labelcolor='#9B2915')
# ax3.set_title('Davies-Bouldin Index (Cool) & Gap Statistic (Warm)', 
#                 fontsize=13, fontweight='bold', pad=15)
# ax3.grid(True, alpha=0.3)
# ax3.legend(loc='upper left', fontsize=8, framealpha=0.9)
# ax3_twin.legend(loc='upper right', fontsize=8, framealpha=0.9)

# # Plot 4: Custom Metric (single axis - use cool colors)
# ax4 = axes[1, 1]

# for algorithm in algorithms:
#     algo_data = dim_data[dim_data['model'] == algorithm].sort_values('clusters')
#     style = get_style(algorithm)
    
#     ax4.plot(algo_data['clusters'], algo_data['aisle_sep'],
#             marker=style['marker'], color=style['color_cool'], linestyle=style['linestyle'],
#             label=algorithm, linewidth=2.5, markersize=8)

# ax4.set_xlabel('Number of Clusters (k)', fontsize=12)
# ax4.set_ylabel('Average Aisle Separation', fontsize=12)
# ax4.set_title('Custom Metric: Average Aisle Separation', fontsize=13, fontweight='bold', pad=15)
# ax4.legend(fontsize=9, framealpha=0.9, loc='best')
# ax4.grid(True, alpha=0.3)

# plt.tight_layout(rect=[0, 0, 1, 0.99], h_pad=3, w_pad=3)
# plt.show()