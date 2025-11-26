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

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap

from collections import defaultdict

loader = CSR_Loader()
umapper = umap.UMAP()
t_embed = TSNE()

data, indices = loader.load_reduced_random("hot_baskets_products", seed=42, n=10000)
# data = loader.load("hot_groceries_baskets")
dimensions = [4, 8, 16, 32, 64, 128]
# dimensions = [2, 4, 6, 8, 10, 12]

variances = []
umap_embeddings = []
tsne_embeddings = []

for k in dimensions:
    print(k)
    tsvd =TruncatedSVD(n_components=k)
    reduced_data = tsvd.fit_transform(data)
    explained_variance = np.sum(tsvd.explained_variance_ratio_)

    variances.append(explained_variance)
    umap_embeddings.append(umapper.fit_transform(reduced_data))
    tsne_embeddings.append(t_embed.fit_transform(reduced_data))

n_col = len(umap_embeddings)
fig, ax = plt.subplots(2, n_col, figsize=(4 * n_col, 8))
fig.suptitle(f"Stocastic DR at Various TSVD Dimensions", 
            fontsize=16, fontweight='bold')

for i in range(n_col):
    umap_data = umap_embeddings[i]
    k_dim = dimensions[i]

    ax[0, i].scatter(umap_data[:, 0], umap_data[:, 1], 
                     s=5,
                     alpha=0.6)
    
    ax[0, i].set_title(f"Explained Variance: {variances[i] * 100:.1f}%\n\n UMAP (TSVD k={k_dim})", fontsize=10, fontweight='bold')
    ax[0, i].set_xlabel("UMAP 1")
    ax[0, i].set_ylabel("UMAP 2")
    ax[0, i].grid(True, alpha=0.3)
    
    tsne_data = tsne_embeddings[i]
    
    ax[1, i].scatter(tsne_data[:, 0], tsne_data[:, 1], 
                     s=5,
                     alpha=0.6)
    
    ax[1, i].set_title(f"t-SNE (TSVD k={k_dim})", fontsize=10, fontweight='bold')
    ax[1, i].set_xlabel("t-SNE 1")
    ax[1, i].set_ylabel("t-SNE 2")
    ax[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


