"""
Affinity Propagation Module 
==================================================
This module implements Affinity Propagation clustering on the Instacart dataset
using various dimensionality reduction techniques.

Requirements:
- parquet_files

Experiments:
1. TSVD + AP on product vectors (4, 8, 16 dimensions)
2. PCA + AP with minimum support filtering (minsup=20)
3. Aisle concatenation BEFORE dimensionality reduction
4. Aisle concatenation AFTER dimensionality reduction

Author: Hetav Raval
Course: DS5230 - Northeastern University
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics.pairwise import euclidean_distances
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CSR LOADER CLASS 
# =============================================================================

class CSR_Loader:
    """
    Loader class for CSR sparse matrices stored as parquet files.
    
    """
    
    def __init__(self, parquet_path="parquet_files"):
        self.path = Path(parquet_path)
        
        # Mapping of data files to their row/column maps
        self.name_map = {
            "hot_baskets_products.parquet": ["hot_map_products.parquet", "hot_map_orders.parquet"],
            "hot_baskets_aisles.parquet":   ["hot_map_aisles.parquet",   "hot_map_orders.parquet"],
            "hot_baskets_depts.parquet":    ["hot_map_depts.parquet",    "hot_map_orders.parquet"],
            
            "hot_customers_products.parquet": ["hot_map_products.parquet", "hot_map_custs.parquet"],
            "hot_customers_aisles.parquet":   ["hot_map_aisles.parquet",   "hot_map_custs.parquet"],
            "hot_customers_depts.parquet":    ["hot_map_depts.parquet",    "hot_map_custs.parquet"],
        }
        
        # Validate that required files exist
        self._validate_files()
    
    def _validate_files(self):
        """Check that required parquet files exist."""
        required_files = [
            "hot_customers_products.parquet",
            "hot_customers_aisles.parquet",
            "hot_baskets_products.parquet",
            "hot_baskets_aisles.parquet",
            "hot_map_products.parquet",
            "hot_map_aisles.parquet",
            "hot_map_orders.parquet",
            "hot_map_custs.parquet",
        ]
        
        missing = []
        for f in required_files:
            if not (self.path / f).exists():
                missing.append(f)
        
        if missing:
            print(f"WARNING: Missing parquet files: {missing}")
            print(f"Download from the SharePoint link and place in '{self.path}/'")
    
    def load(self, filename):
        """
        Load a CSR matrix from parquet file.
        
        Parameters:
        - filename: Name without .parquet extension 
                   (e.g., "hot_customers_products")
        
        Returns:
        - csr_matrix: Sparse matrix
        """
        filename_ext = f"{filename}.parquet"
        
        df = pd.read_parquet(self.path / filename_ext)
        map_a = pd.read_parquet(self.path / self.name_map[filename_ext][1])
        map_b = pd.read_parquet(self.path / self.name_map[filename_ext][0])
        
        coo = coo_matrix(
            (df['value'], (df['row_id'], df['col_id'])),
            shape=(len(map_a), len(map_b))
        )
        
        return coo.tocsr()
    
    def load_reduced_random(self, filename, seed=42, n=10000):
        """
        Load a random subset of the CSR matrix.
        
        Parameters:
        - filename: Name without .parquet extension
        - seed: Random seed for reproducibility (default=42)
        - n: Number of samples (default=10000)
        
        Returns:
        - csr_subset: Sparse matrix subset
        - indices: Original row indices of selected samples
        """
        csr = self.load(filename)
        np.random.seed(seed)
        indices = np.random.choice(csr.shape[0], size=n, replace=False)
        return csr[indices, :], indices
    
    def get_aisles(self, labels, indices, target="hot_customers_aisles"):
        """
        Compute aisle distributions for each cluster.
        
        Parameters:
        - labels: Cluster labels for each sample
        - indices: Original indices of samples
        - target: Aisle matrix to use
        
        Returns:
        - cluster_means: Mean aisle position for each cluster
        - cluster_aisles: Sum of aisle purchases per cluster
        """
        if len(labels) != len(indices):
            raise ValueError("labels and indices must have same length")
        
        matrix = self.load(target)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Create mapping from label to cluster index
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        cluster_pts = [[] for _ in range(n_clusters)]
        
        for i, label in enumerate(labels):
            row = matrix[indices[i]].toarray()
            cluster_pts[label_to_idx[label]].append(row)
        
        cluster_means = []
        cluster_aisles = []
        
        for i in range(n_clusters):
            if len(cluster_pts[i]) > 0:
                cluster_matrix = np.vstack(cluster_pts[i])
                row_sum = cluster_matrix.sum(axis=0).flatten()
                aisle_indices = np.arange(len(row_sum))
                total = row_sum.sum()
                
                cluster_aisles.append(row_sum)
                
                if total > 0:
                    cluster_means.append(np.dot(aisle_indices, row_sum) / total)
                else:
                    cluster_means.append(0)
            else:
                # Empty cluster
                n_aisles = matrix.shape[1]
                cluster_means.append(0)
                cluster_aisles.append(np.zeros(n_aisles))
        
        return cluster_means, cluster_aisles


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_wcss(X, labels, centers=None):
    """
    Compute Within-Cluster Sum of Squares.
    
    """
    wcss = 0
    for label in np.unique(labels):
        cluster_mask = labels == label
        cluster_points = X[cluster_mask]
        
        if centers is not None and label < len(centers):
            center = centers[label]
        else:
            center = cluster_points.mean(axis=0)
        
        wcss += np.sum((cluster_points - center) ** 2)
    
    return wcss


def get_cluster_separations(labels, indices, loader, target="hot_customers_aisles"):
    """
    Compute inter-cluster separation metrics based on aisle distributions.
    
    Returns:
    - avg_separation: Average L2 distance between cluster profiles
    - min_separation: Minimum L2 distance (weakest link)
    - dist_matrix: Full pairwise distance matrix
    """
    means, cluster_aisles = loader.get_aisles(labels, indices, target)
    
    # Normalize aisle distributions
    normalized = []
    for aisles in cluster_aisles:
        total = aisles.sum()
        if total > 0:
            normalized.append(aisles / total)
        else:
            normalized.append(aisles)
    
    if len(normalized) < 2:
        return 0, 0, np.array([[0]]), means
    
    # Compute pairwise distances
    normalized_arr = np.vstack(normalized)
    profiles = normalize(normalized_arr, norm='l1', axis=1)
    dist_matrix = euclidean_distances(profiles, profiles)
    
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)
    
    if mask.sum() > 0:
        avg_separation = np.mean(dist_matrix[mask])
        min_separation = np.min(dist_matrix[mask])
    else:
        avg_separation = 0
        min_separation = 0
    
    return avg_separation, min_separation, dist_matrix, means


def apply_minsup_filter(csr_matrix, min_support=20):
    """
    Filter products by minimum support threshold.
    
    Products appearing fewer than min_support times across
    all customers/baskets are removed.
    
    Parameters:
    - csr_matrix: Sparse matrix (samples x products)
    - min_support: Minimum occurrence threshold
    
    Returns:
    - filtered_matrix: Dense matrix with filtered products
    - kept_indices: Column indices of kept products
    """
    product_counts = np.array(csr_matrix.sum(axis=0)).flatten()
    kept_indices = np.where(product_counts >= min_support)[0]
    filtered_matrix = csr_matrix[:, kept_indices].toarray()
    
    print(f"  MinSup filtering: {csr_matrix.shape[1]} -> {len(kept_indices)} products")
    
    return filtered_matrix, kept_indices


def run_affinity_propagation(X, damping=0.7, preference=None, max_iter=300, random_state=42):
    """
    Run Affinity Propagation clustering.
    
    Parameters:
    - X: Input data (n_samples x n_features)
    - damping: Damping factor [0.5, 1.0) - higher = more stable
    - preference: Controls cluster count (lower = fewer clusters)
    - max_iter: Maximum iterations
    - random_state: Random seed
    
    Returns:
    - ap: Fitted AffinityPropagation model
    - labels: Cluster assignments
    - n_clusters: Number of clusters found
    """
    ap = AffinityPropagation(
        damping=damping,
        preference=preference,
        max_iter=max_iter,
        random_state=random_state,
        affinity='euclidean'
    )
    
    labels = ap.fit_predict(X)
    n_clusters = len(np.unique(labels))
    
    return ap, labels, n_clusters


def compute_all_metrics(X, labels, indices, loader, ap_model=None, aisle_target="hot_customers_aisles"):
    """
    Compute all clustering evaluation metrics.
    
    Metrics:
    - n_clusters: Number of clusters
    - wcss: Within-Cluster Sum of Squares
    - silhouette: Silhouette Score [-1, 1]
    - ch_index: Calinski-Harabasz Index (higher = better)
    - db_index: Davies-Bouldin Index (lower = better)
    - avg_separation: Average aisle profile separation
    - min_separation: Minimum aisle profile separation
    - mean_aisle_variance: Variance of cluster mean aisle positions
    """
    n_clusters = len(np.unique(labels))
    
    # Edge case: single cluster
    if n_clusters < 2:
        return {
            'n_clusters': n_clusters,
            'wcss': np.inf,
            'silhouette': -1,
            'ch_index': 0,
            'db_index': np.inf,
            'avg_separation': 0,
            'min_separation': 0,
            'mean_aisle_variance': 0
        }
    
    # WCSS
    centers = ap_model.cluster_centers_ if ap_model is not None else None
    wcss = compute_wcss(X, labels, centers)
    
    # Silhouette Score
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = -1
    
    # Calinski-Harabasz Index
    try:
        ch_index = calinski_harabasz_score(X, labels)
    except:
        ch_index = 0
    
    # Davies-Bouldin Index
    try:
        db_index = davies_bouldin_score(X, labels)
    except:
        db_index = np.inf
    
    # Aisle-based separation metrics
    try:
        avg_sep, min_sep, _, means = get_cluster_separations(
            labels, indices, loader, aisle_target
        )
        mean_aisle_var = np.var(means) if len(means) > 1 else 0
    except Exception as e:
        print(f"    Warning: Aisle metrics failed - {e}")
        avg_sep, min_sep, mean_aisle_var = 0, 0, 0
    
    return {
        'n_clusters': n_clusters,
        'wcss': wcss,
        'silhouette': silhouette,
        'ch_index': ch_index,
        'db_index': db_index,
        'avg_separation': avg_sep,
        'min_separation': min_sep,
        'mean_aisle_variance': mean_aisle_var
    }


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def experiment_tsvd_ap(loader, damping_vals=[0.5, 0.7, 0.9], pref_pcts=[10, 25, 50]):
    """
    Experiment 1: Affinity Propagation on TSVD-reduced product data.
    
    - Data: hot_customers_products (10,000 samples, seed=42)
    - TSVD dimensions: 4, 8, 16
    - Damping: 0.5, 0.7, 0.9
    - Preference percentiles: 10, 25, 50
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Affinity Propagation on TSVD-reduced Products")
    print("=" * 70)
    
    # Load data with consistent seed and sample size
    data, indices = loader.load_reduced_random(
        filename="hot_customers_products", seed=42, n=10000
    )
    data_norm = normalize(data, norm="l1", axis=1)
    
    TSVD_DIMS = [4, 8, 16]
    results = []
    labels_store = []
    
    for n_comp in TSVD_DIMS:
        print(f"\n--- TSVD d={n_comp} ---")
        
        tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
        X = tsvd.fit_transform(data_norm)
        
        # Compute similarity matrix for preference setting
        similarities = -euclidean_distances(X, squared=True)
        
        for damp in damping_vals:
            for pref_pct in pref_pcts:
                pref = np.percentile(similarities.flatten(), pref_pct)
                
                try:
                    ap, labels, k = run_affinity_propagation(
                        X, damping=damp, preference=pref
                    )
                    
                    # Skip extreme cluster counts
                    if k < 2 or k > 500:
                        print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k} (skipped)")
                        continue
                    
                    metrics = compute_all_metrics(X, labels, indices, loader, ap)
                    
                    result = {
                        "model": "AP_TSVD_products",
                        "tsvd": n_comp,
                        "k": k,
                        "damping": damp,
                        "pref_pct": pref_pct,
                        **metrics
                    }
                    results.append(result)
                    
                    # Store labels in requested format
                    labels_store.append({
                        "model": "AP_TSVD_products",
                        "tsvd": n_comp,
                        "k": k,
                        "labels": labels.tolist(),
                        "WCSS": metrics['wcss']
                    })
                    
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k}, "
                          f"Sil={metrics['silhouette']:.3f}, CH={metrics['ch_index']:.1f}")
                    
                except Exception as e:
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: Error - {e}")
    
    return results, labels_store


def experiment_pca_minsup_ap(loader, min_support=20, 
                             damping_vals=[0.5, 0.7, 0.9], 
                             pref_pcts=[10, 25, 50]):
    """
    Experiment 2: Affinity Propagation on PCA-reduced data with MinSup filtering.
    
    - Data: hot_customers_products (10,000 samples, seed=42)
    - MinSup: 20 
    - PCA dimensions: 4, 8, 16
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Affinity Propagation on PCA with MinSup={min_support}")
    print("=" * 70)
    
    data, indices = loader.load_reduced_random(
        filename="hot_customers_products", seed=42, n=10000
    )
    
    # Apply minimum support filtering
    data_filtered, _ = apply_minsup_filter(data, min_support=min_support)
    data_norm = normalize(data_filtered, norm="l1", axis=1)
    
    PCA_DIMS = [4, 8, 16]
    results = []
    labels_store = []
    
    for n_comp in PCA_DIMS:
        print(f"\n--- PCA d={n_comp} ---")
        
        pca = PCA(n_components=n_comp, random_state=42)
        X = pca.fit_transform(data_norm)
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        similarities = -euclidean_distances(X, squared=True)
        
        for damp in damping_vals:
            for pref_pct in pref_pcts:
                pref = np.percentile(similarities.flatten(), pref_pct)
                
                try:
                    ap, labels, k = run_affinity_propagation(
                        X, damping=damp, preference=pref
                    )
                    
                    if k < 2 or k > 500:
                        print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k} (skipped)")
                        continue
                    
                    metrics = compute_all_metrics(X, labels, indices, loader, ap)
                    
                    result = {
                        "model": "AP_PCA_MinSup_products",
                        "pca": n_comp,
                        "min_support": min_support,
                        "k": k,
                        "damping": damp,
                        "pref_pct": pref_pct,
                        **metrics
                    }
                    results.append(result)
                    
                    labels_store.append({
                        "model": "AP_PCA_MinSup_products",
                        "pca": n_comp,
                        "min_support": min_support,
                        "k": k,
                        "labels": labels.tolist(),
                        "WCSS": metrics['wcss']
                    })
                    
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k}, "
                          f"Sil={metrics['silhouette']:.3f}, CH={metrics['ch_index']:.1f}")
                    
                except Exception as e:
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: Error - {e}")
    
    return results, labels_store


def experiment_aisle_before_dr(loader, damping_vals=[0.5, 0.7, 0.9], pref_pcts=[10, 25, 50]):
    """
    Experiment 3: Aisle matrix concatenated BEFORE dimensionality reduction.
    
    Process:
    1. Load products and aisles matrices
    2. Concatenate: [products | aisles]
    3. Apply TSVD to combined matrix
    4. Run Affinity Propagation
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Aisle Concatenation BEFORE Dimensionality Reduction")
    print("=" * 70)
    
    # Load both matrices with same indices
    products, indices = loader.load_reduced_random(
        filename="hot_customers_products", seed=42, n=10000
    )
    aisles, _ = loader.load_reduced_random(
        filename="hot_customers_aisles", seed=42, n=10000
    )
    
    # Normalize separately
    products_norm = normalize(products, norm="l1", axis=1)
    aisles_norm = normalize(aisles, norm="l1", axis=1)
    
    # Concatenate before DR
    combined = hstack([products_norm, aisles_norm])
    print(f"Combined shape before DR: {combined.shape}")
    
    TSVD_DIMS = [4, 8, 16]
    results = []
    labels_store = []
    
    for n_comp in TSVD_DIMS:
        print(f"\n--- TSVD d={n_comp} (Products+Aisles -> DR) ---")
        
        tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
        X = tsvd.fit_transform(combined)
        
        similarities = -euclidean_distances(X, squared=True)
        
        for damp in damping_vals:
            for pref_pct in pref_pcts:
                pref = np.percentile(similarities.flatten(), pref_pct)
                
                try:
                    ap, labels, k = run_affinity_propagation(
                        X, damping=damp, preference=pref
                    )
                    
                    if k < 2 or k > 500:
                        print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k} (skipped)")
                        continue
                    
                    metrics = compute_all_metrics(X, labels, indices, loader, ap)
                    
                    result = {
                        "model": "AP_Aisle_BeforeDR",
                        "tsvd": n_comp,
                        "k": k,
                        "damping": damp,
                        "pref_pct": pref_pct,
                        **metrics
                    }
                    results.append(result)
                    
                    labels_store.append({
                        "model": "AP_Aisle_BeforeDR",
                        "tsvd": n_comp,
                        "k": k,
                        "labels": labels.tolist(),
                        "WCSS": metrics['wcss']
                    })
                    
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k}, "
                          f"Sil={metrics['silhouette']:.3f}, CH={metrics['ch_index']:.1f}")
                    
                except Exception as e:
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: Error - {e}")
    
    return results, labels_store


def experiment_aisle_after_dr(loader, damping_vals=[0.5, 0.7, 0.9], pref_pcts=[10, 25, 50]):
    """
    Experiment 4: Aisle matrix concatenated AFTER dimensionality reduction.
    
    Process:
    1. Load products matrix, apply TSVD
    2. Load aisles matrix (full 134 dimensions)
    3. Concatenate: [products_reduced | aisles_full]
    4. Run Affinity Propagation
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Aisle Concatenation AFTER Dimensionality Reduction")
    print("=" * 70)
    
    # Load products with random sampling
    products, indices = loader.load_reduced_random(
        filename="hot_customers_products", seed=42, n=10000
    )
    
    # Load full aisles matrix and select same indices
    aisles_full = loader.load("hot_customers_aisles")
    aisles = aisles_full[indices, :]
    
    # Normalize
    products_norm = normalize(products, norm="l1", axis=1)
    aisles_norm = normalize(aisles, norm="l1", axis=1).toarray()
    
    TSVD_DIMS = [4, 8, 16]
    results = []
    labels_store = []
    
    for n_comp in TSVD_DIMS:
        print(f"\n--- TSVD d={n_comp} + Aisles ({aisles_norm.shape[1]}d) ---")
        
        # Apply TSVD to products only
        tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
        products_reduced = tsvd.fit_transform(products_norm)
        
        # Concatenate after DR
        X = np.hstack([products_reduced, aisles_norm])
        print(f"  Combined shape after DR: {X.shape}")
        
        similarities = -euclidean_distances(X, squared=True)
        
        for damp in damping_vals:
            for pref_pct in pref_pcts:
                pref = np.percentile(similarities.flatten(), pref_pct)
                
                try:
                    ap, labels, k = run_affinity_propagation(
                        X, damping=damp, preference=pref
                    )
                    
                    if k < 2 or k > 500:
                        print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k} (skipped)")
                        continue
                    
                    metrics = compute_all_metrics(X, labels, indices, loader, ap)
                    
                    result = {
                        "model": "AP_Aisle_AfterDR",
                        "tsvd": n_comp,
                        "total_dims": X.shape[1],
                        "k": k,
                        "damping": damp,
                        "pref_pct": pref_pct,
                        **metrics
                    }
                    results.append(result)
                    
                    labels_store.append({
                        "model": "AP_Aisle_AfterDR",
                        "tsvd": n_comp,
                        "k": k,
                        "labels": labels.tolist(),
                        "WCSS": metrics['wcss']
                    })
                    
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k}, "
                          f"Sil={metrics['silhouette']:.3f}, CH={metrics['ch_index']:.1f}")
                    
                except Exception as e:
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: Error - {e}")
    
    return results, labels_store


# =============================================================================
# BASKET-LEVEL EXPERIMENTS
# =============================================================================

def experiment_tsvd_ap_baskets(loader, damping_vals=[0.5, 0.7, 0.9], pref_pcts=[10, 25, 50]):
    """
    Experiment 5: AP on TSVD-reduced basket (order) data.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Affinity Propagation on TSVD-reduced Baskets")
    print("=" * 70)
    
    data, indices = loader.load_reduced_random(
        filename="hot_baskets_products", seed=42, n=10000
    )
    data_norm = normalize(data, norm="l1", axis=1)
    
    TSVD_DIMS = [4, 8, 16]
    results = []
    labels_store = []
    
    for n_comp in TSVD_DIMS:
        print(f"\n--- TSVD d={n_comp} (Baskets) ---")
        
        tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
        X = tsvd.fit_transform(data_norm)
        
        similarities = -euclidean_distances(X, squared=True)
        
        for damp in damping_vals:
            for pref_pct in pref_pcts:
                pref = np.percentile(similarities.flatten(), pref_pct)
                
                try:
                    ap, labels, k = run_affinity_propagation(
                        X, damping=damp, preference=pref
                    )
                    
                    if k < 2 or k > 500:
                        continue
                    
                    # Use basket-level aisle target
                    metrics = compute_all_metrics(
                        X, labels, indices, loader, ap,
                        aisle_target="hot_baskets_aisles"
                    )
                    
                    result = {
                        "model": "AP_TSVD_baskets",
                        "tsvd": n_comp,
                        "k": k,
                        "damping": damp,
                        "pref_pct": pref_pct,
                        **metrics
                    }
                    results.append(result)
                    
                    labels_store.append({
                        "model": "AP_TSVD_baskets",
                        "tsvd": n_comp,
                        "k": k,
                        "labels": labels.tolist(),
                        "WCSS": metrics['wcss']
                    })
                    
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k}, "
                          f"Sil={metrics['silhouette']:.3f}")
                    
                except Exception as e:
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: Error - {e}")
    
    return results, labels_store


def experiment_aisle_before_dr_baskets(loader, damping_vals=[0.5, 0.7, 0.9], pref_pcts=[10, 25, 50]):
    """
    Experiment 6: Aisle concatenation before DR on basket data.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Aisle Before DR on Baskets")
    print("=" * 70)
    
    products, indices = loader.load_reduced_random(
        filename="hot_baskets_products", seed=42, n=10000
    )
    aisles, _ = loader.load_reduced_random(
        filename="hot_baskets_aisles", seed=42, n=10000
    )
    
    products_norm = normalize(products, norm="l1", axis=1)
    aisles_norm = normalize(aisles, norm="l1", axis=1)
    
    combined = hstack([products_norm, aisles_norm])
    print(f"Combined shape before DR: {combined.shape}")
    
    TSVD_DIMS = [4, 8, 16]
    results = []
    labels_store = []
    
    for n_comp in TSVD_DIMS:
        print(f"\n--- TSVD d={n_comp} (Baskets+Aisles) ---")
        
        tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
        X = tsvd.fit_transform(combined)
        
        similarities = -euclidean_distances(X, squared=True)
        
        for damp in damping_vals:
            for pref_pct in pref_pcts:
                pref = np.percentile(similarities.flatten(), pref_pct)
                
                try:
                    ap, labels, k = run_affinity_propagation(
                        X, damping=damp, preference=pref
                    )
                    
                    if k < 2 or k > 500:
                        continue
                    
                    metrics = compute_all_metrics(
                        X, labels, indices, loader, ap,
                        aisle_target="hot_baskets_aisles"
                    )
                    
                    result = {
                        "model": "AP_Aisle_BeforeDR_baskets",
                        "tsvd": n_comp,
                        "k": k,
                        "damping": damp,
                        "pref_pct": pref_pct,
                        **metrics
                    }
                    results.append(result)
                    
                    labels_store.append({
                        "model": "AP_Aisle_BeforeDR_baskets",
                        "tsvd": n_comp,
                        "k": k,
                        "labels": labels.tolist(),
                        "WCSS": metrics['wcss']
                    })
                    
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k}, "
                          f"Sil={metrics['silhouette']:.3f}")
                    
                except Exception as e:
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: Error - {e}")
    
    return results, labels_store


def experiment_aisle_after_dr_baskets(loader, damping_vals=[0.5, 0.7, 0.9], pref_pcts=[10, 25, 50]):
    """
    Experiment 7: Aisle concatenation after DR on basket data.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 7: Aisle After DR on Baskets")
    print("=" * 70)
    
    products, indices = loader.load_reduced_random(
        filename="hot_baskets_products", seed=42, n=10000
    )
    aisles_full = loader.load("hot_baskets_aisles")
    aisles = aisles_full[indices, :]
    
    products_norm = normalize(products, norm="l1", axis=1)
    aisles_norm = normalize(aisles, norm="l1", axis=1).toarray()
    
    TSVD_DIMS = [4, 8, 16]
    results = []
    labels_store = []
    
    for n_comp in TSVD_DIMS:
        print(f"\n--- TSVD d={n_comp} + Aisles ({aisles_norm.shape[1]}d) (Baskets) ---")
        
        tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
        products_reduced = tsvd.fit_transform(products_norm)
        
        X = np.hstack([products_reduced, aisles_norm])
        print(f"  Combined shape after DR: {X.shape}")
        
        similarities = -euclidean_distances(X, squared=True)
        
        for damp in damping_vals:
            for pref_pct in pref_pcts:
                pref = np.percentile(similarities.flatten(), pref_pct)
                
                try:
                    ap, labels, k = run_affinity_propagation(
                        X, damping=damp, preference=pref
                    )
                    
                    if k < 2 or k > 500:
                        continue
                    
                    metrics = compute_all_metrics(
                        X, labels, indices, loader, ap,
                        aisle_target="hot_baskets_aisles"
                    )
                    
                    result = {
                        "model": "AP_Aisle_AfterDR_baskets",
                        "tsvd": n_comp,
                        "total_dims": X.shape[1],
                        "k": k,
                        "damping": damp,
                        "pref_pct": pref_pct,
                        **metrics
                    }
                    results.append(result)
                    
                    labels_store.append({
                        "model": "AP_Aisle_AfterDR_baskets",
                        "tsvd": n_comp,
                        "k": k,
                        "labels": labels.tolist(),
                        "WCSS": metrics['wcss']
                    })
                    
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: k={k}, "
                          f"Sil={metrics['silhouette']:.3f}")
                    
                except Exception as e:
                    print(f"  damp={damp:.1f}, pref_pct={pref_pct}: Error - {e}")
    
    return results, labels_store


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_all_experiments(loader, include_baskets=True):
    """
    Run all Affinity Propagation experiments.
    
    Parameters:
    - loader: CSR_Loader instance
    - include_baskets: Whether to run basket-level experiments
    
    Returns:
    - results_df: DataFrame with all metrics
    - labels_list: List of label dictionaries
    """
    all_results = []
    all_labels = []
    
    # Customer-level experiments
    print("\n" + "#" * 80)
    print("# CUSTOMER-LEVEL EXPERIMENTS")
    print("#" * 80)
    
    r1, l1 = experiment_tsvd_ap(loader)
    all_results.extend(r1)
    all_labels.extend(l1)
    
    r2, l2 = experiment_pca_minsup_ap(loader, min_support=20)
    all_results.extend(r2)
    all_labels.extend(l2)
    
    r3, l3 = experiment_aisle_before_dr(loader)
    all_results.extend(r3)
    all_labels.extend(l3)
    
    r4, l4 = experiment_aisle_after_dr(loader)
    all_results.extend(r4)
    all_labels.extend(l4)
    
    # Basket-level experiments
    if include_baskets:
        print("\n" + "#" * 80)
        print("# BASKET-LEVEL EXPERIMENTS")
        print("#" * 80)
        
        r5, l5 = experiment_tsvd_ap_baskets(loader)
        all_results.extend(r5)
        all_labels.extend(l5)
        
        r6, l6 = experiment_aisle_before_dr_baskets(loader)
        all_results.extend(r6)
        all_labels.extend(l6)
        
        r7, l7 = experiment_aisle_after_dr_baskets(loader)
        all_results.extend(r7)
        all_labels.extend(l7)
    
    results_df = pd.DataFrame(all_results)
    
    return results_df, all_labels


def save_results(results_df, labels_list, prefix="ap"):
    """
    Save results in multiple formats.
    
    Output files:
    - {prefix}_metrics.csv: All metrics
    - {prefix}_labels.json: Labels in JSON format
    - {prefix}_labels.csv: Labels in CSV format
    """
    # Metrics CSV
    results_df.to_csv(f"{prefix}_metrics.csv", index=False)
    print(f"\nSaved: {prefix}_metrics.csv")
    
    # Labels JSON
    with open(f"{prefix}_labels.json", "w") as f:
        json.dump(labels_list, f, indent=2)
    print(f"Saved: {prefix}_labels.json")
    
    # Labels CSV (for integration with visualization code)
    labels_df = pd.DataFrame([
        {
            "model": v.get("model", ""),
            "tsvd": v.get("tsvd", ""),
            "pca": v.get("pca", ""),
            "k": v.get("k", ""),
            "WCSS": v.get("WCSS", ""),
            "labels": json.dumps(v.get("labels", []))
        }
        for v in labels_list
    ])
    labels_df.to_csv(f"{prefix}_labels.csv", index=False)
    print(f"Saved: {prefix}_labels.csv")


def print_summary(results_df):
    """Print experiment summary."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal experiments completed: {len(results_df)}")
    
    if len(results_df) > 0:
        print("\n--- Results by Model ---")
        summary = results_df.groupby('model').agg({
            'k': ['min', 'max', 'mean'],
            'silhouette': ['max', 'mean'],
            'ch_index': ['max', 'mean'],
            'db_index': ['min', 'mean']
        }).round(3)
        print(summary)
        
        print("\n--- Top 5 by Silhouette Score ---")
        cols = ['model', 'k', 'silhouette', 'ch_index', 'db_index', 'damping']
        available_cols = [c for c in cols if c in results_df.columns]
        best_sil = results_df.nlargest(5, 'silhouette')[available_cols]
        print(best_sil.to_string(index=False))
        
        print("\n--- Top 5 by Calinski-Harabasz Index ---")
        best_ch = results_df.nlargest(5, 'ch_index')[available_cols]
        print(best_ch.to_string(index=False))
        
        print("\n--- Top 5 by Davies-Bouldin Index (lower is better) ---")
        best_db = results_df.nsmallest(5, 'db_index')[available_cols]
        print(best_db.to_string(index=False))


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AFFINITY PROPAGATION CLUSTERING - GMO Final Project")
    print("=" * 80)
    print("\nRequirements:")
    print("  - parquet_files/ directory with pre-processed matrices")
    print("=" * 80)
    
    # Initialize loader
    loader = CSR_Loader(parquet_path="parquet_files")
    
    # Run all experiments
    results_df, labels_list = run_all_experiments(loader, include_baskets=True)
    
    # Save results
    save_results(results_df, labels_list, prefix="ap")
    
    # Print summary
    print_summary(results_df)
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
