from pathlib import Path
import sys

current_file = Path(__file__).resolve()
gmo_final_path = current_file.parent.parent
sys.path.append(str(gmo_final_path))

import numpy as np
from matplotlib import pyplot as plt


from instacart_loader import CSR_Loader

from sklearn.decomposition import TruncatedSVD

from collections import defaultdict

loader = CSR_Loader()

data1, indices = loader.load_reduced_random("hot_baskets_products", seed=42, n=10000)
data2 = loader.load("hot_groceries_baskets")
data3, indices = loader.load_reduced_random("hot_baskets_products", seed=42, n=100000)
data4 = loader.load("hot_baskets_products")
datas = [data2, data1, data3, data4]
names = ["Groceries", "InstaCart-10k", "InstaCart-100k", "InstaCart-All"]

k = 4

depth = k // 2
embeddings = [[] for _ in range(depth)]
for data in datas:
    tsvd =TruncatedSVD(n_components=k)
    reduced_data = tsvd.fit_transform(data)
    print(reduced_data.shape)
    for d in range(depth):
        idx = d * 2
        embeddings[d].append(np.stack((reduced_data[:, idx], reduced_data[:, idx + 1]), axis=1))


if len(datas) >= depth:
    n_col = len(datas)
    n_row = len(embeddings)
    fig, ax = plt.subplots(2, n_col, figsize=(2 * n_col, 2 * n_row))
    fig.suptitle(f"Dimension Plots for {k}D Data", 
                fontsize=16, fontweight='bold')

    for i in range(n_col):
        for j in range(n_row):
            embed_data = embeddings[j][i]
            category_name = names[i]
            p = (j + 1) * 2
            dim = (p-1, p)
            
            ax[j, i].scatter(embed_data[:, 0], embed_data[:, 1], 
                            s=5,
                            alpha=0.6)
            
            ax[j, i].set_title(f"Dim {dim[0]} vs Dim {dim[1]} for {category_name}", 
                            fontsize=10, fontweight='bold')
            ax[j, i].set_xlabel(f"Component {dim[0]}")
            ax[j, i].set_ylabel(f"Component {dim[1]}")
            ax[j, i].grid(True, alpha=0.3)
else:
    n_col = len(embeddings)
    n_row = len(datas)
    fig, ax = plt.subplots(n_row, n_col, figsize=(2 * n_col, 2 * n_row))
    fig.suptitle(f"Dimension Plots for {k}D Data", 
                fontsize=16, fontweight='bold')
    
    for i in range(n_row):
        for j in range(n_col):
            embed_data = embeddings[j][i]
            if j == 0:
                category_name = names[i]
            else:
                category_name = ""
            p = (j + 1) * 2
            dim = (p-1, p)
            
            ax[i, j].scatter(embed_data[:, 0], embed_data[:, 1], 
                            s=5,
                            alpha=0.6)
            
            ax[i, j].set_title(f"{dim[0]} vs {dim[1]}", 
                            fontsize=10, fontweight='bold')
            ax[i, j].set_xlabel(f"C{dim[0]}")
            ax[i, j].set_ylabel(f"{category_name} C{dim[1]}")
            ax[i, j].grid(True, alpha=0.3)


plt.tight_layout()
plt.show()