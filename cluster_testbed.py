from instacart_loader import CSR_Loader
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt




l = CSR_Loader()
pc = PCA(n_components=6) #from sindico
km = KMeans(n_clusters=4)

'''discrepancies in plotting are due to '''

aisle_matrix, indices = l.load_reduced_random(filename="hot_customers_aisles", seed=42, n=10000)
# aisle_matrix = l.load("hot_customers_aisles")

aisle_matrix = normalize(aisle_matrix, norm="l1", axis=1)
am_reduced = pc.fit_transform(aisle_matrix)[:, [4, 1]]
labels = km.fit_predict(am_reduced)

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
# plt.show()

means, cluster_aisles = l.get_aisle_means(labels, indices)

for i in range(len(cluster_aisles)):
    cluster_aisles[i] = np.sqrt(cluster_aisles[i] / cluster_aisles[i].sum())

print(means, np.var(means))





cluster_matrix = np.vstack(cluster_aisles)
cluster_profiles = normalize(cluster_matrix, norm='l1', axis=1)
dist_matrix = euclidean_distances(cluster_profiles, cluster_profiles)
cluster_profiles = normalize(cluster_matrix, norm='l1', axis=1)
dist_matrix = euclidean_distances(cluster_profiles, cluster_profiles)

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



# Assuming 'dist_matrix' is the one you calculated in the previous step
# We use a mask to ignore the diagonal (0) and the duplicates (lower triangle)
mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)

# Calculate the mean of just the valid distances
avg_separation = np.mean(dist_matrix[mask])
min_separation = np.min(dist_matrix[mask])

print(f"Average Distance between Cluster Centers: {avg_separation:.4f}")
print(f"Minimum Distance (The 'Weakest Link'):    {min_separation:.4f}")



















plt.figure(figsize=(15, 8))

# Loop through each cluster
for i, counts in enumerate(cluster_aisles):
    
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

plt.xticks(np.arange(0, 135, 10))
plt.xlim(0, 134)             # Ensure the shading starts/ends cleanly
plt.ylim(bottom=0)           # Start Y-axis at 0

plt.show()