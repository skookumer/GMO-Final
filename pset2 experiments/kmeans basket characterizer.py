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

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from collections import defaultdict

loader = CSR_Loader()



data, indices = loader.load_reduced_random("hot_baskets_products", seed=42, n=10000)

clusterer = KMeans(n_clusters=30, random_state=42, max_iter=10000, algorithm="elkan")

tsvd = TruncatedSVD(n_components=4)

data = tsvd.fit_transform(data)

labels = clusterer.fit_predict(data)

unique = np.unique(labels).tolist()
items = []

for i in range(len(unique)):
    cluster_id = unique[i]
    labels_idx = [j for j in range(len(labels)) if labels[j] == cluster_id]
    order_ids = [indices[idx] for idx in labels_idx]
    if len(order_ids) > 1:
        dist = loader.retrieve_target_information(order_ids, "hot_baskets_products", names=True)

        unique_items = {}
        for basket in dist:
            for item in basket:
                if item in unique_items:
                    unique_items[item] += 1
                else:
                    unique_items[item] = 1

        unique_items_sorted = sorted(unique_items.items(), key=lambda x: x[1], reverse=True)
        top = unique_items_sorted[:100]

        print(top)

        # Extract items and counts
        items = [item for item, count in top]
        counts = [count for item, count in top]

        # Create histogram/bar plot
        plt.figure(figsize=(20, 6))
        sns.barplot(x=items, y=counts)
        plt.xticks(rotation=90, ha='right')
        plt.xlabel('Item')
        plt.ylabel('Frequency')
        plt.title('Top 100 Items by Frequency')
        plt.tight_layout()
        plt.show()





