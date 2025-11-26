#code to let this run from the subfolder
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
gmo_final_path = current_file.parent.parent
sys.path.append(str(gmo_final_path))

"""
DS5230 - Group Project 2
Group 4
HAC (centroid linkage) on the Groceries dataset.
Author: Zachary Merriam
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# Data loading and preprocessing
def load_groceries_data(csv_path: str) -> pd.DataFrame:
    """
    Load the Groceries dataset from a CSV file and construct basket IDs.
    Parameters
    ----------
    csv_path : str
        Path to Groceries_dataset.csv.
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ['basket_id', 'item'].
    """
    df = pd.read_csv(csv_path)

    # Builds basket identifiers. 
    if "Member_number" in df.columns and "Date" in df.columns:
        df["basket_id"] = (
            df["Member_number"].astype(str) + "_" + df["Date"].astype(str)
        )
    elif "Transaction" in df.columns:
        df["basket_id"] = df["Transaction"]
    else:
        # Fallback: treat each row as its own basket.
        df["basket_id"] = np.arange(len(df))

    # Standardize the item column to 'item'.
    if "itemDescription" in df.columns:
        df = df.rename(columns={"itemDescription": "item"})
    elif "item" not in df.columns:
        raise ValueError(
            "Could not find an item column (expected 'itemDescription' or 'item')."
        )
    return df[["basket_id", "item"]]


def build_item_basket_matrix(df: pd.DataFrame, top_n: int = 60):
    """
    Build an item-by-basket binary matrix for the top-N frequent items.
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['basket_id', 'item'].
    top_n : int
        Number of most frequent items to keep.
    Returns
    -------
    item_names : list of str
        Names of the retained items (length = top_n).
    X : numpy.ndarray, shape (top_n, n_baskets)
        Entry (i, j) is 1 if item i appears in basket j, 0 otherwise.
    """
    item_counts = df["item"].value_counts()
    item_names = item_counts.head(top_n).index.tolist()

    df_top = df[df["item"].isin(item_names)].copy()

    basket_ids = df_top["basket_id"].unique()
    basket_to_idx = {b: i for i, b in enumerate(basket_ids)}
    item_to_idx = {item: i for i, item in enumerate(item_names)}

    n_items = len(item_names)
    n_baskets = len(basket_ids)

    X = np.zeros((n_items, n_baskets), dtype=float)

    for _, row in df_top.iterrows():
        i = item_to_idx[row["item"]]
        j = basket_to_idx[row["basket_id"]]
        X[i, j] = 1.0

    return item_names, X

# Clustering and validation
def run_hac_centroid(X: np.ndarray, k: int):
    """
    Run Hierarchical Agglomerative Clustering (centroid linkage).
    Parameters
    ----------
    X : numpy.ndarray, shape (n_items, n_features)
        Data matrix where each row is an item representation.
    k : int
        Number of clusters.
    Returns
    -------
    labels : numpy.ndarray, shape (n_items,)
        Cluster labels in {1, ..., k}.
    Z : numpy.ndarray
        Linkage matrix returned by scipy.
    """
    Z = linkage(X, method="centroid", metric="euclidean")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels, Z

def compute_wcss(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute within-cluster sum of squares (WCSS) for HAC clusters.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_items, n_features)
    labels : numpy.ndarray, shape (n_items,)

    Returns
    -------
    wcss : float
    """
    wcss = 0.0
    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        if len(cluster_points) == 0:
            continue
        center = cluster_points.mean(axis=0, keepdims=True)
        wcss += np.sum((cluster_points - center) ** 2)
    return float(wcss)


# Visualization 
def plot_pca_clusters(X, labels, item_names, out_path):
    """
    Project items to 2D using PCA and plot clusters with nicer spacing
    so labels don't get cut off at the figure edges.
    Parameters
    ----------
    X : numpy.ndarray, shape (n_items, n_features)
        Item representations.
    labels : numpy.ndarray, shape (n_items,)
        Cluster labels.
    item_names : list of str
        Names of the items.
    out_path : str
        Output path for the PNG file.
    """
    # PCA projection to 2D 
    pca = PCA(n_components=2, random_state=0)
    X_2d = pca.fit_transform(X)

    xs = X_2d[:, 0]
    ys = X_2d[:, 1]

    # Calculate padding based on data range
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    x_pad = 0.1 * x_range if x_range > 0 else 1.0
    y_pad = 0.1 * y_range if y_range > 0 else 1.0

    plt.figure(figsize=(11, 7))
    scatter = plt.scatter(xs, ys, c=labels, s=50, cmap="tab10")

    # Label placement logic to avoid clipping
    base_dx = 0.02 * x_range if x_range > 0 else 0.2
    base_dy = 0.02 * y_range if y_range > 0 else 0.2

    x_max = xs.max()
    x_min = xs.min()

    for i, name in enumerate(item_names):
        x, y = xs[i], ys[i]

        # If point is near the right edge, pull the label left; if near left, push right
        if x > x_max - 0.05 * x_range:
            dx = -base_dx
            ha = "right"
        elif x < x_min + 0.05 * x_range:
            dx = base_dx
            ha = "left"
        else:
            dx = base_dx
            ha = "left"

        plt.text(
            x + dx,
            y + base_dy,
            name,
            fontsize=7,
            ha=ha,
            va="bottom",
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Groceries - HAC (centroid) - PCA projection")

    # Apply the padded limits
    plt.xlim(xs.min() - x_pad, xs.max() + x_pad)
    plt.ylim(ys.min() - y_pad, ys.max() + y_pad)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_tsne_clusters(X, labels, item_names, out_path, perplexity=15, random_state=0):
    """
    t-SNE projection of items with label placement that avoids clipping.
    Parameters
    ----------
    X : numpy.ndarray, shape (n_items, n_features)
        Item representations.
    labels : numpy.ndarray, shape (n_items,)
        Cluster labels.
    item_names : list of str
        Names of the items.
    out_path : str
        Output path for the PNG file.
    perplexity : float
        t-SNE perplexity parameter.
    random_state : int
        Random seed for t-SNE.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=random_state,
    )
    # 2D t-SNE projection 
    X_2d = tsne.fit_transform(X)

    xs = X_2d[:, 0]
    ys = X_2d[:, 1]

    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    x_pad = 0.1 * x_range if x_range > 0 else 1.0
    y_pad = 0.1 * y_range if y_range > 0 else 1.0

    plt.figure(figsize=(11, 7))
    plt.scatter(xs, ys, c=labels, s=50, cmap="tab10")

    base_dx = 0.02 * x_range if x_range > 0 else 0.2
    base_dy = 0.02 * y_range if y_range > 0 else 0.2

    x_max = xs.max()
    x_min = xs.min()
    y_max = ys.max()
    y_min = ys.min()

    # Label placement logic to avoid clipping
    for i, name in enumerate(item_names):
        x, y = xs[i], ys[i]

        # Horizontal position of label
        if x > x_max - 0.05 * x_range:
            dx = -base_dx
            ha = "right"
        elif x < x_min + 0.05 * x_range:
            dx = base_dx
            ha = "left"
        else:
            dx = base_dx
            ha = "left"

        # Vertical position of label
        if y > y_max - 0.05 * y_range:
            dy = -base_dy
            va = "top"
        elif y < y_min + 0.05 * y_range:
            dy = base_dy
            va = "bottom"
        else:
            dy = base_dy
            va = "bottom"

        plt.text(
            x + dx,
            y + dy,
            name,
            fontsize=7,
            ha=ha,
            va=va,
        )

    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.title("Groceries - HAC (centroid) - t-SNE projection")

    plt.xlim(xs.min() - x_pad, xs.max() + x_pad)
    plt.ylim(ys.min() - y_pad, ys.max() + y_pad)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_dendrogram(Z, item_names, out_path, max_items=60):
    """
    Plot a dendrogram for the HAC result.
    Parameters
    ----------
    Z : numpy.ndarray
        Linkage matrix from scipy.
    item_names : list of str
        Names of the items in the same order as X.
    out_path : str
        Output path for the PNG file.
    max_items : int
        Maximum number of leaf labels to draw.
    """
    plt.figure(figsize=(14, 7))
    dendrogram(
        Z,
        labels=item_names if len(item_names) <= max_items else item_names[:max_items],
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=None,
    )
    plt.title("Groceries - HAC (centroid) dendrogram")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_elbow(metrics_df, out_path):
    """
    Plot WCSS vs k (elbow curve).
    Parameters
    ----------
    metrics_df : pandas.DataFrame
        DataFrame with columns including 'k' and 'wcss'.
    out_path : str
        Output path for PNG.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_df["k"], metrics_df["wcss"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Within-cluster sum of squares (WCSS)")
    plt.title("Groceries - HAC (centroid) - elbow curve")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# Main script
def main():
    groceries_path = gmo_final_path / "instacart_data" / "Groceries_dataset.csv"
    top_n = 60
    ks = [3, 4, 5, 6, 7]

    use_svd = True # Whether to reduce dimensionality with SVD before clustering. 
    svd_components = 4 # Number of SVD components if used. Chosen to align with instacart experiments.

    print("Loading Groceries data from:", groceries_path)
    df = load_groceries_data(groceries_path)
    print(f"Number of raw rows: {len(df)}")

    print(f"Building item-basket matrix for top {top_n} items...")
    item_names, X_raw = build_item_basket_matrix(df, top_n=top_n)
    print("Matrix shape (items x baskets):", X_raw.shape)

    # SVD dimensionality reduction
    if use_svd:
        print(f"Running TruncatedSVD to {svd_components} dimensions...")
        svd = TruncatedSVD(n_components=svd_components, random_state=0)
        X = svd.fit_transform(X_raw)
        print("Shape after SVD:", X.shape)
    else:
        X = X_raw

    # Hyperparameter sweep over k + validation metrics
    print("\nRunning HAC (centroid linkage) for k =", ks)

    metrics_records = []

    # Sweep over k values and compute validation metrics 
    for k in ks:
        start = time.time()
        labels, Z = run_hac_centroid(X, k=k)
        runtime = time.time() - start

        # Validation metrics – items as points
        sil = silhouette_score(X, labels, metric="euclidean")
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        wcss = compute_wcss(X, labels)

    # Store metrics for this k value 
        metrics_records.append(
            {
                "k": k,
                "silhouette": sil,
                "davies_bouldin": db,
                "calinski_harabasz": ch,
                "wcss": wcss,
                "runtime_sec": runtime,
            }
        )

        print(
            f"k = {k:2d} | silhouette = {sil:6.3f} "
            f"| Davies-Bouldin = {db:6.3f} "
            f"| Calinski-Harabasz = {ch:6.1f} "
            f"| WCSS = {wcss:8.1f} "
            f"| runtime = {runtime:5.3f} s"
        )
    # Compile metrics into DataFrame
    metrics_df = pd.DataFrame(metrics_records)

    # Save metrics for later comparison with Instacart results.
    metrics_path = "groceries_hac_centroid_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved validation metrics to: {metrics_path}")

    # Elbow plot
    figs_dir = "figs"
    os.makedirs(figs_dir, exist_ok=True)
    elbow_path = os.path.join(figs_dir, "groceries_hac_centroid_elbow.png")
    plot_elbow(metrics_df, elbow_path)
    print("Saved elbow curve to:", elbow_path)

    # Here I keep k=5 to stay consistent with the group discussion.
    best_k = 5
    print(f"\nUsing k = {best_k} for cluster interpretation and plots...")
    labels_best, Z_best = run_hac_centroid(X, k=best_k)

    # Print items by cluster membership.
    print("\nCluster memberships for top items (k = 5):")
    for cluster_id in range(1, best_k + 1):
        cluster_items = [
            name for name, lab in zip(item_names, labels_best) if lab == cluster_id
        ]
        print(f"\nCluster {cluster_id} ({len(cluster_items)} items):")
        print(", ".join(cluster_items))

    # PCA / t-SNE / dendrogram
    pca_path = os.path.join(figs_dir, "groceries_hac_centroid_pca.png")
    tsne_path = os.path.join(figs_dir, "groceries_hac_centroid_tsne.png")
    dendro_path = os.path.join(figs_dir, "groceries_hac_centroid_dendrogram.png")

    print("\nSaving PCA scatter plot to:", pca_path)
    plot_pca_clusters(X, labels_best, item_names, pca_path)

    print("Saving t-SNE scatter plot to:", tsne_path)
    plot_tsne_clusters(X, labels_best, item_names, tsne_path)

    print("Saving dendrogram plot to:", dendro_path)
    plot_dendrogram(Z_best, item_names, dendro_path)

if __name__ == "__main__":
    main()
