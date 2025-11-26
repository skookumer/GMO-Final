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
aisle_data = loader.load("hot_baskets_aisles")

ks = [i * 10 for i in range(1, 11)]
ds = [2 ** i for i in range(1, 8)]

# ks = [i * 10 for i in range(1, 5)]
# ds = [2 * i for i in range(1, 4)]

results = []

for k in ks:

    clusterer = KMeans(n_clusters=k, random_state=42, max_iter=10000, algorithm="elkan")
    for d in ds:
        print(k, d)
        tsvd = TruncatedSVD(n_components=d)

        data_reduced = tsvd.fit_transform(data)
        labels = clusterer.fit_predict(data_reduced)
        unique = np.unique(labels).tolist()

        entropies = []

        for i in range(len(unique)):
            cluster_id = unique[i]
            labels_idx = [j for j in range(len(labels)) if labels[j] == cluster_id]
            order_ids = [indices[idx] for idx in labels_idx]
            if len(order_ids) > 1:
                dist = np.full(134, 1e-10)
                for order_id in order_ids:
                    row = aisle_data[order_id, :]
                    row = row.todense()
                    dist += np.array(row)[0]
                dist /= np.sum(dist)

                entropy = 0
                for j in range(134):
                    entropy += dist[j] * np.log2(dist[j])
                entropies.append(-entropy)

        results.append({"d": d, "k": k, "ent": np.mean(entropies)})

df = pd.DataFrame(results)
        
sns.lineplot(
    data=df,
    x="k",
    y="ent",
    hue="d",
    marker="o",
    palette="Set1",
    legend="full"
)

plt.title("InstaCart 100k: Avg Aisle Entropy for varying d and k")
plt.xlabel("Number of Clusters")
plt.ylabel("Mean Entropy")

plt.legend(title=r'Dimensions ($\text{d}$)')
plt.xticks(sorted(df['k'].unique()))

plt.grid(True, linestyle='--', alpha=0.6)
plt.show()