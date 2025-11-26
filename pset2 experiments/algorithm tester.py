from pathlib import Path
import sys

current_file = Path(__file__).resolve()
gmo_final_path = current_file.parent.parent
sys.path.append(str(gmo_final_path))


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import time
import tracemalloc

from instacart_loader import CSR_Loader
from instacart_loader import process_parquet

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.cluster import KMeans, AffinityPropagation
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from pyclustering.cluster import clique


import umap
from pathlib import Path
import psutil, os
import gc

class hac_centroid:

    def __init__(self, k):

        self.data = data
        self.k = k
        self.n_iter_ = 1
    
    def fit_predict(self, data):
        z = linkage(data, method="centroid")
        labels = fcluster(z, self.k, criterion="maxclust")
        return labels
    
def calculate_wcss(data, labels):
    wcss = 0
    unique = np.unique(labels)
    
    for label in unique:
        if label == -1:
            continue
        pts = data[labels == label]
        mu = pts.mean(axis=0)
        wcss += np.sum((pts - mu) ** 2)
    
    return wcss


def plot_algorithm(axis, target, x_lab, y_lab, title, log=False):
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo].sort_values('k')
        if len(algo_data) > 0:
            if not log:
                axis.plot(algo_data['k'], algo_data[target], 
                        marker='o', label=algo, color=color_map[algo], 
                        linewidth=2, markersize=6, alpha=0.8)
            else:
                axis.plot(algo_data['k'], np.log(algo_data[target]), 
                        marker='o', label=algo, color=color_map[algo], 
                        linewidth=2, markersize=6, alpha=0.8)
    axis.set_xlabel(x_lab, fontsize=12)
    axis.set_ylabel(y_lab, fontsize=12)
    axis.set_title(title, fontsize=13, fontweight='bold')
    axis.legend(fontsize=9, loc='best')
    axis.grid(True, alpha=0.3)
    

process = psutil.Process(os.getpid())
print(f"Starting: {process.memory_info().rss / 1024**3:.1f} GB")
# process_parquet()
loader = CSR_Loader()


                                
data, indices = loader.load_reduced_random("hot_baskets_products", seed=42, n=10000) #This is for loading data from instacart
# data = loader.load("hot_groceries_baskets") #FOR GROCERIES

# data , top_names = loader.get_cooccurrence_matrix(k=120)

# pca = PCA(n_components=5)
# data = pca.fit_transform(data)
tsvd = TruncatedSVD(n_components=4)
data = tsvd.fit_transform(data)
# data = data.astype(np.float32)

# data = np.array(data.todense())

ks = [k * 10 for k in range(1, 11)] #SET THIS FOR INTERVALS OF K

# ks=[5, 10]

'''********TSNE COMES AT THE END. YOU HAVE TO ENTER PERPLEXITY AND THE VALUE OF K MANUALLY********'''

results = []
for k in ks:
    algorithms = {
        'AffinityPropagation': AffinityPropagation(damping=0.9, random_state=42, max_iter=10000), #Takes a very long time
        'KMeans': KMeans(n_clusters=k, random_state=42, max_iter=10000, algorithm="elkan"),
        'GMM': GaussianMixture(n_components=k, max_iter=10000, random_state=42),
        'HAC Centroid': hac_centroid(k=k),
        # 'CLIQUE': clique(data=data, amount_intervals=k/2),
        'BGMM prior=0.1': BayesianGaussianMixture(n_components=k, weight_concentration_prior=0.1, random_state=42, max_iter=10000),
        'BGMM prior=1': BayesianGaussianMixture(n_components=k, weight_concentration_prior=1, random_state=42, max_iter=10000),
        'BGMM prior=100': BayesianGaussianMixture(n_components=k, weight_concentration_prior=100, random_state=42, max_iter=10000),
    }
    for name, model in algorithms.items():

        print(name, k)

        if k > ks[0] and name == "AffinityPropagation":
            result = results[0].copy()
            result["k"] = k
            results.append(result)
        else:
            start = time.time()
            tracemalloc.start()
            if hasattr(model, 'fit_predict'):
                labels = model.fit_predict(data)
            else:  # For GMM and BGMM
                model.fit(data)
                labels = model.predict(data)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            end = time.time()
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            results.append({
                "k": k,
                "Algorithm": name,
                "Clusters": n_clusters,
                "Silhouette": silhouette_score(data, labels),
                "Davies-Bouldin": davies_bouldin_score(data, labels),
                "Calinski-Harabasz": calinski_harabasz_score(data, labels),
                "WCSS": calculate_wcss(data, labels),
                "Labels": labels,
                "Time": end - start,
                "Memory_MB": peak / 1024**2,
                "Iterations": np.log(model.n_iter_)
            })

df = pd.DataFrame(results)


k_based_algos = ['KMeans', "HAC Centroid", 'GMM', 'BGMM prior=0.1', 'BGMM prior=1', 'BGMM prior=100']
auto_algos = ['AffinityPropagation']

color_map = {
    'KMeans': 'blue',
    "HAC Centroid": 'cyan',
    'GMM': 'green',
    'BGMM prior=0.1': 'red',
    'BGMM prior=1': 'orange',
    'BGMM prior=100': 'purple',
    'AffinityPropagation': 'brown'
}


fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Clustering Algorithm Performance Across k Values', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
plot_algorithm(ax1, "Silhouette", "Number of Clusters", "Silhouette", "Silhouette Score vs k")

ax2 = axes[0, 1]
plot_algorithm(ax2, "Davies-Bouldin", "Number of Clusters", "D-B Index", "Davies-Bouldin Index vs k")

ax3 = axes[1, 0]
plot_algorithm(ax3, "Calinski-Harabasz", "Number of Clusters", "C-H Index", "Calinski-Harabasz Index vs k")

ax4 = axes[1, 1]
plot_algorithm(ax4, "WCSS", "Number of clusters", "WCSS", "WCSS vs k")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Algorithm Performance: Time and Memory Usage', fontsize=16, fontweight='bold')

ax1 = axes[0]
plot_algorithm(ax1, "Time", "Number of Clusters", "Execution Time (log scale) (seconds)", "Execution Time vs k", log=True)

ax2 = axes[1]
plot_algorithm(ax2, "Memory_MB", "Number of Clusters", "Peak Memory (log scale) (MB)", "Memory Usage vs k", log=True)

plt.tight_layout()
plt.show()

fig2, ax = plt.subplots(figsize=(10, 8))
plot_algorithm(ax, "Clusters", "k", "Clusters Found", "Actual vs Expected Number of Clusters")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Iterations to Convergence (log scale)', fontsize=16, fontweight='bold')

plot_algorithm(ax, "Iterations", "Number of Clusters", "Iterations to Convergence (log scale)", "Iterations vs k")



while True:

    perplex = input("enter perplexity ")

    reducer = TSNE(perplexity=int(perplex), random_state=42)
    # umapper = umap.UMAP(n_neighbors=10, min_dist=.01, random_state=42)
    data_2d = reducer.fit_transform(data)

    plot_k = input("enter desired k ")
    while (int(plot_k)) not in ks:
        plot_k = input("enter desired k ")

    k_to_plot = int(plot_k)

    # Filter results for this k value
    k_results = df[df['k'] == k_to_plot]
    algorithms = k_results['Algorithm'].unique()

    n_algorithms = len(algorithms)
    n_cols = int(np.ceil(np.sqrt(n_algorithms)))
    n_rows = int(np.ceil(n_algorithms / n_cols))

    print(f"Creating {n_rows}x{n_cols} grid for {n_algorithms} algorithms")

    # Create adaptive subplot grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    fig.suptitle(f'Cluster Assignments for k={k_to_plot} (TSNE Visualization)', 
                fontsize=16, fontweight='bold')

    # Handle different cases for axes
    if n_algorithms == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    else:
        axes_flat = axes.flatten()

    # Plot each algorithm
    for idx, algo in enumerate(algorithms):
        ax = axes_flat[idx]
        
        # Get labels for this algorithm
        algo_row = k_results[k_results['Algorithm'] == algo].iloc[0]
        labels = algo_row['Labels']
        n_clusters = algo_row['Clusters']
        
        # Create scatter plot
        scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                            c=labels, cmap='tab10', s=30, alpha=0.7, 
                            edgecolors='black', linewidths=0.3)
        
        # Add title with metrics
        ax.set_title(f"{algo}\n"
                    f"Clusters: {n_clusters} | Silhouette: {algo_row['Silhouette']:.3f}\n"
                    f"Time: {algo_row['Time']:.2f}s | Iterations: {round(np.exp(algo_row['Iterations']))}", 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('TSNE 1', fontsize=9)
        ax.set_ylabel('TSNE 2', fontsize=9)
        ax.grid(True, alpha=0.2)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Cluster')

    # Hide any unused subplots
    for idx in range(n_algorithms, n_rows * n_cols):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.show()

        
        



bgmm = BayesianGaussianMixture(n_components=7, weight_concentration_prior=0.1, random_state=42)
# bgmm = GaussianMixture(n_components=7, random_state=42)
bgmm.fit(data_2d)
labels = bgmm.predict(data_2d)
print(len(labels))

print(f"Mixture weights: {bgmm.weights_}")
# Organize names by cluster
clusters_dict = {}
for cluster_id in range(bgmm.n_components):
    # Find which points belong to this cluster
    cluster_mask = (labels == cluster_id)
    
    # Get the names of those points
    cluster_names = top_names[cluster_mask]
    
    # Store in dictionary
    clusters_dict[cluster_id] = cluster_names.tolist()

# Print organized clusters
for cluster_id, names in clusters_dict.items():
    if len(names) > 0:  # Only print non-empty clusters
        print(f"\nCluster {cluster_id} ({len(names)} products):")
        for name in names:
            print(f"  - {name}")

plt.figure(figsize=(10, 8))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, alpha=0.5, cmap='viridis')
plt.title('Bayesian GMM Clusters')
plt.xlabel('UMAP dimension 1')
plt.ylabel('UMAP dimension 2')
plt.colorbar(label='Cluster')
plt.show()