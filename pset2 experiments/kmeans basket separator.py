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

from scipy.spatial.distance import euclidean
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from collections import defaultdict

loader = CSR_Loader()

data, indices = loader.load_reduced_random("hot_baskets_products", seed=42, n=10000)
data = loader.load("hot_groceries_baskets")
indices = np.array([i for i in range(data.shape[0])])

n = 100

clusterer = KMeans(n_clusters=20, random_state=42, max_iter=10000, algorithm="elkan")
# tsvd = TruncatedSVD(n_components=167)

# data = tsvd.fit_transform(data)
data = np.array(data.todense())
labels = clusterer.fit_predict(data)

centroids = {}
unique = np.unique(labels).tolist()
for k in unique:
    cluster_points = data[labels == k]
    centroids[k] = {"coord": np.mean(cluster_points, axis=0), "size": len(cluster_points)}
centroids = list(centroids.values())

dmax = -1
furthest = (None, None)
for i in range(len(unique)):
    lab_i = unique[i]
    c_i = centroids[lab_i]["coord"]
    c_i_size = centroids[lab_i]["size"]
    for j in range(i + 1, len(unique)):
        lab_j = unique[j]
        c_j = centroids[lab_j]["coord"]
        c_j_size = centroids[lab_j]["size"]
        distance = euclidean(c_i, c_j) * (np.log(np.min([c_i_size, c_j_size]))) #scaling the distance by the log of the min to take into account size
        if distance > dmax:
            dmax = distance
            furthest = (lab_i, lab_j)


items = []

for i in range(len(furthest)):
    cluster_id = furthest[i]
    labels_idx = [j for j in range(len(labels)) if labels[j] == cluster_id]
    order_ids = [indices[idx] for idx in labels_idx]
    if len(order_ids) > 1:
        dist = loader.retrieve_target_information(order_ids, "hot_baskets_products", names=True)

        total_items = 0
        unique_items = {}
        for basket in dist: #gather up items
            for item in basket:
                if item in unique_items:
                    unique_items[item] += 1
                else:
                    unique_items[item] = 1
                total_items += 1
        
        # for key in unique_items: #normalize
        #     unique_items[key] /= total_items

        items.append(unique_items.copy())

union = []
for key in items[0]:
    if key in items[1]:
        x = items[0][key]
        y = items[1][key]
        union.append({"item": key, 
                      f"{furthest[0]}_count": x, 
                      f"{furthest[1]}_count": y, 
                      "abs_diff": (abs(x - y) / (x + y)) * np.log(x + y)}) #chose to normalize here instead. weighted by the log sum to encourage frequency

df = pd.DataFrame(union)

df_top = df.sort_values(by='abs_diff', ascending=False).head(n)
plot_height = max(6, len(df_top) * 0.3) 

fig, ax = plt.subplots(figsize=(plot_height, 10))
ind = np.arange(len(df_top))
width = 0.35

bar1 = ax.bar(ind, df_top[f"{furthest[0]}_count"], 
              label=f"{furthest[0]}_count", color='#4c72b0', alpha=0.9)
bar2 = ax.bar(ind, df_top[f"{furthest[1]}_count"], 
              bottom=df_top[f"{furthest[0]}_count"], 
              label=f"{furthest[1]}_count", color='#dd8452', alpha=0.9)


ax.set_title(f'Groceries-167d: Differences between most different clusters', 
                fontsize=14, pad=15)
ax.set_xlabel('Item Count', fontsize=12)
ax.set_xticks(ind)
ax.set_xticklabels(df_top['item'], rotation=90, ha='center')

# max_count = df_top[[f"{furthest[0]}_count", f"{furthest[1]}_count"]].values.max()

# for i, row in df_top.iterrows():
#     ax.text(max_count * 1.05, ind[i] + 0.1, 
#             f'Diff: {row["abs_diff"]}', 
#             va='center', 
#             color='gray', 
#             fontsize=7,
#             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))


# ax.legend()
ax.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()