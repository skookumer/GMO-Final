import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os
import random

from scipy.sparse import spmatrix
from sklearn.metrics import (                       
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD as TSVD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from itertools import combinations, chain
from functools import reduce
import json
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import ast

from instacart_loader import CSR_Loader

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
def convert_to_numpy(obj):
    """Recursively convert lists to numpy arrays and ints to np.int64"""
    if isinstance(obj, dict):
        return {k: convert_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Check if it's a list of lists (nested structure to preserve)
        if obj and isinstance(obj[0], list):
            # Convert each inner list to a numpy array
            return [np.array(item) for item in obj]
        # Empty or numeric list - convert to array
        if not obj or isinstance(obj[0], (int, float)):
            return np.array(obj)
        return [convert_to_numpy(item) for item in obj]
    elif isinstance(obj, int):
        return np.int64(obj)
    elif isinstance(obj, float):
        return np.float64(obj)
    return obj

class Gen_Optimizer:

    def __init__(self, sys_username="Eric Arnold", smoothing_alpha=1, 
                 cycle_limit=2, combi_limit=3, pop_cap=1000, mode="rulemine", 
                 toprint=False, uniform=False, k_range=(2, 10), mut_rate=0.1, n_components=8, minsup=20, metrics=["sup", "conf"]):
        self.cycle_limit = cycle_limit
        self.combi_limit = combi_limit
        self.username = sys_username
        self.pop_cap = pop_cap
        self.mode = mode
        self.toprint = toprint
        self.uniform = uniform
        self.k_range=k_range
        self.mut_rate = mut_rate
        self.minsup = minsup


        self.high_item_ids = [24849, 13173, 21134, 21900, 47205, 47762, 47622, 16794, 26206, 27842, 27963, 22932, 24961, 45004, 39272, 49679, 28201, 5874, 40703, 8274, 4918, 30388, 45063, 42262, 44629, 49231, 19054, 4603, 21613, 37643, 17791, 27101, 30486, 31714, 27083, 46975, 8515, 44356, 28982, 41947, 26601, 5075, 34123, 22032, 39874, 43349, 35948, 10746, 19657, 9073, 24181, 21935, 43958, 34966, 48675, 46663, 12338, 25887, 31503, 5448, 39925, 24835, 22822, 5783, 35218, 28839, 33728, 8421, 27518, 33195, 8171, 44139, 20111, 27341, 11517, 29484, 18462, 28196, 15287, 46902, 9836, 27153, 3955, 43119, 23906, 34355, 4797, 9384, 16756, 195, 42733, 4208, 38686, 41784, 47140, 41217, 7778, 32997, 20992, 21706]

        self.m1 = {}
        self.m2 = {}
        self.memo_filenames = ["memo_1.jsonl", "memo_2.jsonl"]

        if platform.system() == "Windows":
            self.memo_path = Path("C:/") / "Users" / self.username / "Documents" / "NSGA_data"
        elif platform.system() == "Darwin":
            self.memo_path = Path.home() / "Documents" / "NSGA_data"

        self.loader = CSR_Loader()
        self.filename = "hot_baskets_products"
        self.path = Path(__file__).parent / "parquet_files"

        if mode in ["testmode", "rulemine"]:
            dense_name = f"{self.filename}_tidset_dense.parquet"
            self.metrics = metrics
            self.memo_path = self.memo_path / "miner"
            if not os.path.isfile(self.path / dense_name):
                print("initializing matrices")

                matrix = self.loader.load(self.filename)#[:10000]
                matrix_T = matrix.transpose()
                matrix_T = matrix_T.tocsr()

                indices = matrix_T.indices
                indptr = matrix_T.indptr
                n_rows = matrix_T.shape[0]

                tidset_dense = []
                for i in range(n_rows):
                    start_idx = indptr[i]
                    end_idx = indptr[i+1]

                    tids = indices[start_idx:end_idx]
                    tidset_dense.append({"product_id": i, "tids": tids, "len": len(tids)})
                
                dense_df = pd.DataFrame(tidset_dense)
                dense_df.to_parquet(self.path / dense_name)

            tidset_dense = pd.read_parquet(self.path / dense_name)
            self.tidset_dense = tidset_dense.set_index("product_id")
            tidset_sparse = self.loader.load(self.filename)#[:10000]
            self.N = tidset_sparse.sum()

            sups = np.array((tidset_sparse > 0).sum(axis=0)).flatten()
            minsup_cols = np.where(sups >= minsup)[0]

            self.tidset_sparse = tidset_sparse.tocsc()
            self.pid_map = pd.read_parquet(self.path / "hot_map_products.parquet")
            pdist = np.array(tidset_sparse.sum(axis=0)).flatten() / self.N
            pdist += smoothing_alpha
            # self.pdist = pdist / pdist.sum() #matching the size of pdist based on minsup
            self.pdist = pdist[minsup_cols] / pdist[minsup_cols].sum()
            self.item_id_range = minsup_cols #tidset_sparse.shape[1]

        elif mode in ["testclustomers", "clustomers"]:
            self.metrics = metrics
            self.memo_path = self.memo_path / "clusterer"
            
            print("initializing matrices")

            # matrix = self.loader.load(self.filename)
            matrix, indices = self.loader.load_reduced_random("hot_customers_products", seed=42, n=10000)
            self.cid_matrix_indices = indices
            svd = TSVD(n_components=n_components)
            matrix = normalize(matrix, norm="l1", axis=1)
            self.cid_matrix = svd.fit_transform(matrix)
            self.cid_bounds = [(self.cid_matrix[:, i].min(), self.cid_matrix[:, i].max()) for i in range(self.cid_matrix.shape[1])]
            self.aid_matrix = self.loader.load("hot_customers_aisles")[indices, :]
            self.did_matrix = self.loader.load("hot_customers_depts")[indices, :]
            if mode == "testclustomers":
                self.cid_range = 250
            elif mode == "clustomers":
                self.cid_range = self.cid_matrix.shape[0]

        if mode == "testmode":
            self.items = self.high_item_ids
            self.pdist = np.full(len(self.high_item_ids), 1) / len(self.high_item_ids)
            self.uniform = True
            self.testmode = True
        elif mode == "rulemine":
            self.items = self.item_id_range
            self.testmode = False
        elif mode in ["testclustomers", "clustomers"]:
            self.items = self.cid_range
            self.testmode = False

        if not self.memo_path.exists():
            self.memo_path.mkdir(parents=True, exist_ok=True)


    '''RULE MINING----------------------------------------------------'''
    def get_itemset_support_basic(self, ivec, names=False):
        '''
        this retrieves the smallest product from the dense dataframe to get specfific indices
        to check in the CSR matrix columns. This avoids having to iterate through the whole sparse
        matrix
        '''
        if names:
            pid_series = self.pid_map.set_index("col_id").loc[product_ids, "product_id"]
            names = self.loader.product_names_df.loc[pid_series, 'product_name']

        if isinstance(ivec, spmatrix):
            product_ids = ivec.indices
        else:
            product_ids = ivec
        
        iset_df = self.tidset_dense.loc[product_ids, ["tids", "len"]]

        if len(product_ids) > 1:
            min_prod = iset_df["len"].idxmin()
            tids = iset_df.loc[min_prod, "tids"]

            #try:
            sparse_rows = self.tidset_sparse[:, product_ids]
            sparse_filtered = sparse_rows[tids, :]
            # except:
            #     print("SIMPLE SPARSE ERROR\n", product_ids, tids, iset_df)
            #     input()
            binary_intersection = sparse_filtered.sign()
            cap = np.where(binary_intersection.getnnz(axis=1) == binary_intersection.shape[1])[0]
            if len(cap) > 0:
                # counts = sparse_filtered[cap, :].sum(axis=1).A1
                counts = sparse_filtered[cap, :].min(axis=1).A.flatten()
                filtered_tids = tids[cap]
                return product_ids, np.array(filtered_tids), np.array(counts)
            else:
                return product_ids, [], []
        else:
            tids = iset_df["tids"].iloc[0]
            sparse_rows = self.tidset_sparse[:, product_ids]
            sparse_filtered = sparse_rows[tids, :]
            return product_ids, tids, sparse_filtered.A.flatten()


    def update_memo_2(self, key, tids=[], counts=[]):
        tids = np.array(tids)
        counts = np.array(counts)
        sort_idx = np.argsort(tids)
        self.m2[key] = {
                        "tids": np.array(tids)[sort_idx], 
                        "counts": np.array(counts)[sort_idx], 
                        "sum": int(counts.sum()), 
                        "cycle": self.cycle
                    }


    def get_itemset_support(self, pfx_key, ivec, names=False):
        '''
        this retrieves the smallest product from the dense dataframe to get specfific indices
        to check in the CSR matrix columns. This avoids having to iterate through the whole sparse
        matrix
        '''
        if names:
            pid_series = self.pid_map.set_index("col_id").loc[product_ids, "product_id"]
            names = self.loader.product_names_df.loc[pid_series, 'product_name']
            print(names)

        if isinstance(ivec, spmatrix):
            product_ids = ivec.indices
        else:
            product_ids = ivec


        if len(pfx_key) > 0 and len(product_ids) > 0:
            iset_df = self.tidset_dense.loc[product_ids, ["tids", "len"]]
            pfx = self.m2[pfx_key]
            key_all = tuple(sorted(product_ids + pfx_key))

            min_prod = iset_df["len"].idxmin()
            tids = np.intersect1d(iset_df.loc[min_prod, "tids"], pfx["tids"])
            pfx_counts = pfx["counts"][np.searchsorted(pfx["tids"], tids)]

            sparse_rows = self.tidset_sparse[:, product_ids]
            sparse_filtered = sparse_rows[tids, :]
            binary_intersection = sparse_filtered.sign()
            cap = np.where(binary_intersection.getnnz(axis=1) == binary_intersection.shape[1])[0]
            if len(cap) > 0:
                # iset_counts = sparse_filtered[cap, :].sum(axis=1).A1
                counts = np.minimum(sparse_filtered[cap, :].min(axis=1).A.flatten(), pfx_counts[cap])
                filtered_tids = tids[cap]
                self.update_memo_2(key_all, filtered_tids, counts)
            else:
                self.update_memo_2(key_all)
        elif len(pfx_key) > 0 and len(product_ids) == 0:
            pfx = self.m2[pfx_key]
            self.update_memo_2(pfx, np.array(pfx["tids"]), np.array(pfx["counts"]).min(axis=1))
        elif len(pfx_key) == 0 and len(product_ids) > 1:
            product_ids, tids, counts = self.get_itemset_support_basic(product_ids)
            self.update_memo_2(product_ids, tids, counts)
        else:
            try:
                iset_df = self.tidset_dense.loc[product_ids, ["tids", "len"]]
                tids = iset_df["tids"].iloc[0]
                sparse_rows = self.tidset_sparse[:, product_ids]
                sparse_filtered = sparse_rows[tids, :]
                self.update_memo_2(product_ids, tids, sparse_filtered.A.flatten())
            except:
                print("FAILED")
                print(product_ids)
                print(iset_df)
                input()


    def get_metrics(self, ivec, i):

        '''
        use combos specified in i and find the highest-confidence rule in the itemset.
        This rule is assumed to be definitive of the itemset
        '''

        if isinstance(ivec, spmatrix):
            product_ids = ivec.indices
        else:
            product_ids = ivec
        
        ants = {}
        for r in range(min(i, len(product_ids) - 1), 0, -1):
            combos = combinations(product_ids, r)
            for ant_key in combos:
                if ant_key not in self.m2:
                    self.get_itemset_support(tuple(), ant_key)
                ants[ant_key] = self.m2[ant_key]["sum"]
        sup_all = self.m2[product_ids]["sum"]
        ant_key, sup_ant = min(ants.items(), key=lambda x: x[1]) #select the lowest_support antecedent to get highest-confidence rule
        coq_key = tuple(x for x in product_ids if x not in set(ant_key))
        if coq_key not in self.m2:
            self.get_itemset_support(tuple(), coq_key)
        sup_coq = self.m2[coq_key]["sum"]

        if sup_all != 0 and sup_ant != 0 and sup_coq != 0:

            # ind_sups = []
            # for i in range(len(product_ids)):
            #     ind_sups.append(self.m2[product_ids[i]])
            
            # x = sum(ind_sups)
            # sup_ent = -np.sum([(sup / x) * np.log2(sup / x)  for sup in ind_sups]) / np.log2(len(ind_sups))


            self.m1[product_ids] = {"tag": (ant_key, coq_key),
                                    "sup": sup_all,
                                    # "sup_ent": sup_ent,
                                    "jacc": sup_all / (sup_ant + sup_coq - sup_all),
                                    "conf": (sup_all / sup_ant),
                                    "kulc": (sup_all * (sup_ant + sup_coq)) / (2 * sup_ant * sup_coq),
                                    "lift": (sup_all * self.N) / (sup_coq * sup_ant), 
                                    "length": len(product_ids),
                                    "cycle": self.cycle
                                    }


    def compute_itemset_fitness(self, ivec, i):

        '''
        prefix-enabled miner that attempts to use solved combinations as much as possible
        '''

        def intersect_memoized(k1, k2):
            cap_name = tuple(sorted(k1 + k2))
            t1 = self.m2[k1]
            t2 = self.m2[k2]
            tids1 = t1["tids"]
            tids2 = t2["tids"]
            self.m2[k1]["cycle"] = self.cycle
            self.m2[k1]["cycle"] = self.cycle

            cap_tids = np.intersect1d(tids1, tids2)

            if len(cap_tids) > 0:
                try:
                    cap_counts = np.minimum(t1["counts"][np.searchsorted(tids1, cap_tids)],
                                            t2["counts"][np.searchsorted(tids2, cap_tids)])
                except:
                    print("PREFIX COMBI FAILURE", k1, k2, "\n")
                    print(f"tids1 (len={len(tids1)}): {tids1}")
                    print(f"tids2 (len={len(tids2)}): {tids2}")
                    print(f"counts1 (len={len(t1['counts'])}): {t1['counts']}")
                    print(f"counts2 (len={len(t2['counts'])}): {t2['counts']}")
                    print(f"cap_tids (len={len(cap_tids)}): {cap_tids}")
                    print(f"tids1 sorted? {np.all(tids1[:-1] <= tids1[1:])}")
                    print(f"tids2 sorted? {np.all(tids2[:-1] <= tids2[1:])}")

                    print("\nTable 1:")
                    print(pd.DataFrame({"tids": tids1, "counts": t1["counts"]}))
                    print("\nTable 2:")
                    print(pd.DataFrame({"tids": tids2, "counts": t2["counts"]}))
                    input()
            else:
                cap_counts = np.array([])
            self.update_memo_2(cap_name, cap_tids, cap_counts)
            return cap_name

        if isinstance(ivec, spmatrix):
            key = ivec.indices
        else:
            key = ivec
        
        #check if fitness has been calculated
        if key in self.m1:
            return
        
        #check if any subsets have been computed
        original_key = key
        subset_tids = []
        key = list(key)
        if len(key) > 1 and len(key) <= 10:
            k = i #reduced this to i=3 because worried about combinatorial explosion
            while  k > 0 and len(key) > 1:
                found = False
                subsets = combinations(key, k)
                for subset in subsets:
                    if subset in self.m2:
                        if len(self.m2[subset]["tids"]) > 0:
                            subset_tids.append(subset)
                            for item in subset:
                                key.remove(item)
                            found = True
                            break
                if found == False:
                    k -= 1
        key = tuple(key)
        
        if len(subset_tids) > 0:
            prefix = reduce(intersect_memoized, subset_tids)
        else:
            prefix = tuple()

        if self.toprint:
            print("P", prefix, "K", key, "ORIG", original_key)

        if len(key) > 0:
            self.get_itemset_support(prefix, key)
        if len(original_key) > 1:
            self.get_metrics(original_key, i)


    '''FILE HANDLING----------------------------------------------------'''
    def load_memos(self, u=2):

        memos = [self.m1, self.m2]
        for i in range(u):
            if os.path.exists(self.memo_path / self.memo_filenames[i]):
                print("loading", self.memo_filenames[i])
                with open(self.memo_path / self.memo_filenames[i]) as f:
                    for line in f:
                        entry = json.loads(line)
                        for str_key, value in entry.items():
                            key = tuple(np.int64(x) for x in ast.literal_eval(str_key))
                            converted_value = convert_to_numpy(value)
                            memos[i][key] = converted_value
                print("loaded", self.memo_filenames[i])

    def save_memos(self):
        
        memos = [self.m1, self.m2]
        for i in range(2):
            with open(self.memo_path / self.memo_filenames[i], "w") as f:
                for key, value in memos[i].items():
                    if self.mode in ["clustomers", "testclustomers"]:
                        entry = {str(tuple(item for sublist in key for item in sublist)): value}
                    else:
                        entry = {str(tuple(int(x) for x in key)): value}
                    f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")
    
    def prune_memos(self):

        memos = [self.m1, self.m2]
        for i in range(2):
            to_pop = []
            for key in memos[i]:
                if memos[i][key]["cycle"] <= self.cycle - self.cycle_limit:
                    to_pop.append(key)

            for key in to_pop:
                memos[i].pop(key)

    def load_popn(self, return_obj=True):

        filepath = self.memo_path / "popn.jsonl"

        if os.path.exists(filepath):
            print("load popn from saved")
            if return_obj:
                with open(self.memo_path / "popn.jsonl") as f:
                    line = deque(f, maxlen=1)[0]
                    line = json.loads(line)
                    self.popn = [tuple(individual) for individual in line["population_array"]]
                    self.cycle = line["cycle"]
                    print("loaded cycle", self.cycle)
            else:
                population = []
                cycles = []
                with open(self.memo_path / "popn.jsonl") as f:
                    for line in f.readlines():
                        line = json.loads(line)
                        popn = [tuple(individual) for individual in line["population_array"]]
                        cycle = line["cycle"]
                        population.append(popn.copy())
                        cycles.append(cycle)
                return population, cycles
        else:
            print("generating popn")
            self.cycle = 1
            if self.mode in ["testmode", "rulemine"]:
                self.popn = self.init_popn()
            else:
                self.popn = self.init_clusters()
            self.save_popn()
    
    def save_popn(self):
        print("\nsaving popn")
        
        filepath = self.memo_path / "popn.jsonl"
        mode = ["w", "a"]

        if os.path.exists(filepath):
            mode = mode[1]
        else:
            mode = mode[0]

        with open(self.memo_path / "popn.jsonl", mode) as f:
            f.write(json.dumps({"population_array": self.popn, "cycle": self.cycle}, cls=NumpyEncoder) + "\n")





    '''NSGA STUFF----------------------------------------------------'''
    def init_popn(self):

        population = []
        for _ in range(self.pop_cap):
            k = np.random.choice(self.k_range)
            if self.uniform:
                ind = tuple(sorted(np.random.choice(self.items, k, replace=False, p=self.pdist)))
                pids, tids, counts = self.get_itemset_support_basic(ind)
                while np.sum(counts) < self.minsup:
                    ind = tuple(sorted(np.random.choice(self.items, k, replace=False, p=self.pdist)))
                    pids, tids, counts  = self.get_itemset_support_basic(ind)
                print(pids, np.sum(counts))
                self.update_memo_2(ind, tids, counts)
            else:
                ind = tuple(sorted(np.random.choice(self.items, k, replace=False, p=self.pdist)))
            population.append(ind)
        return population
    
    def init_clusters(self):
        '''distinction between itemsets and clusters: all customerids must be accounted for without replacement'''

        def make_cluster(pool):
            if self.toprint:
                print("making cluster", len(pool))
            if len(pool) == 0:
                return None
            if self.uniform:
                n = np.random.choice(range(1, len(pool) + 1))
            else:
                n = np.min([np.random.choice(range(1, len(pool) + 1)), np.random.poisson(lam=(self.cid_range // self.k_range))])
            cluster = [np.random.choice(list(pool), n, replace=False)]
            next_cluster = make_cluster(pool - set(cluster[0]))
            if next_cluster != None:
                cluster.extend(next_cluster)
            return cluster

        pool = set(range(0, self.cid_range))

        samples = []
        for _ in range(self.pop_cap):
            samples.append(make_cluster(pool))

        self.popn = samples


    def _crowding_distance_vectorized(self, front, n):
        '''Vectorized crowding distance computation'''
        
        if len(front) <= 2:
            return front[:n]
        
        n_individuals = len(front)
        distances = np.zeros(n_individuals)
        
        # Get metrics for this front
        front_metrics = np.array([[self.m1[k][metric] for metric in self.metrics] 
                                for k in front])
        
        for m_idx in range(len(self.metrics)):
            # Sort indices by this metric
            sorted_idx = np.argsort(front_metrics[:, m_idx])
            
            # Boundary points get infinite distance
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            # Compute range
            metric_range = front_metrics[sorted_idx[-1], m_idx] - front_metrics[sorted_idx[0], m_idx]
            
            if metric_range > 0:
                # Vectorized distance computation for middle points
                distances[sorted_idx[1:-1]] += (
                    (front_metrics[sorted_idx[2:], m_idx] - front_metrics[sorted_idx[:-2], m_idx]) 
                    / metric_range
                )
        
        # Select top n individuals by crowding distance
        top_n_idx = np.argsort(distances)[-n:][::-1]
        return [front[i] for i in top_n_idx]
    

    def select_survivors(self, return_fittest=False):

        '''vectorized version of select survivors'''


        m1_keys = list(self.m1.keys())
        n_pop = len(m1_keys)
        metrics_array = np.array([[self.m1[k][metric] for metric in self.metrics] 
                                for k in m1_keys])
        
        if self.toprint:
            print("computing dominance")
        greater = metrics_array[:, None, :] > metrics_array[None, :, :]
        dominates = np.all(greater, axis=2)  # shape: (n_pop, n_pop)
        np.fill_diagonal(dominates, False)  # individual doesn't dominate itself
        dom_counts = np.sum(dominates, axis=0).astype(int)

        fronts = []
        remaining = np.ones(n_pop, dtype=bool)
        
        if self.toprint:
            print("sorting non-dominated")
        while remaining.any():
            # Find individuals not dominated by any remaining individual
            current_front_mask = (dom_counts == 0) & remaining
            if not current_front_mask.any():
                break
                
            current_front = np.where(current_front_mask)[0]
            fronts.append([m1_keys[i] for i in current_front])
            
            # Remove current front and update domination counts
            remaining[current_front] = False
            dom_counts -= dominates[current_front, :].sum(axis=0).astype(int)
            dom_counts[current_front] = -1   # Mark as processed
        
        if self.toprint:
            print("selecting survivors")
        # === SELECT SURVIVORS ===
        sorted_individuals = []
        if return_fittest:
            return fronts[0]
        for front in fronts:
            if len(sorted_individuals) + len(front) <= self.pop_cap:
                sorted_individuals.extend(front)
            else:
                # Need crowding distance for last front
                k = self.pop_cap - len(sorted_individuals)
                rem = self._crowding_distance_vectorized(front, k)
                sorted_individuals.extend(rem)
                break
        if self.toprint:
            print("survivors selected")
    
        return sorted_individuals

    
    def select_survivors_v1(self):

        '''find fronts, comput crowding distance for last front up to length self.pop_cap'''

        if self.toprint:
            print("selecting survovors")

        m = len(self.metrics)

        def is_dom(p, q):
            if sum([int(self.m1[p][self.metrics[x]] > self.m1[q][self.metrics[x]]) for x in range(m)]) == m:
                dom_keys[p].add(q)
                dom_counts[q] += 1
            elif sum([int(self.m1[q][self.metrics[x]] > self.m1[p][self.metrics[x]]) for x in range(m)]) == m:
                dom_keys[q].add(p)
                dom_counts[p] += 1
        
        def crowding_distance(n, front):
            
            if len(front) <= 2:
                return {k: np.inf for k in front}
            
            dists = {k:0 for k in front}
            for metric in self.metrics:
                sf = sorted(front, key = lambda x: self.m1[x][metric])
                dists[sf[0]] = np.inf
                dists[sf[-1]] = np.inf

                mrange = self.m1[sf[-1]][metric] - self.m1[sf[0]][metric]

                if mrange != 0:
                    for i in range(1, len(sf) - 1):
                        if dists[sf[i]] != np.inf:
                            p = self.m1[sf[i + 1]][metric]
                            q = self.m1[sf[i - 1]][metric]
                            dists[sf[i]] += (p - q) / mrange
            
            sorted_dists = sorted(dists.items(), key = lambda x: x[1], reverse=True)
            return [k for k, dist in sorted_dists[:n]]
        

        m1_keys = list(self.m1.keys())

        fronts =[[]]
        dom_keys = defaultdict(set)
        dom_counts = {k:0 for k in self.m1}
        for i in range(len(self.m1)):
            p = m1_keys[i]
            for j in range(i, len(self.m1)):
                if i != j:
                    q = m1_keys[j]
                    is_dom(p, q)
            if dom_counts[p] == 0:
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            front = []
            for p in fronts[i]:
                for q in dom_keys[p]:
                    dom_counts[q] -= 1
                    if dom_counts[q] == 0:
                        front.append(q)
            i += 1
            fronts.append(front.copy())
        
        sorted_individuals = []
        for i in range(len(fronts)):
            if len(sorted_individuals) + len(fronts[i]) < self.pop_cap:
                sorted_individuals.extend(fronts[i])
            else:
                k = self.pop_cap - len(sorted_individuals)
                rem = crowding_distance(k, fronts[i])
                sorted_individuals.extend(rem)
                break
        return sorted_individuals
    


    def compute_cluster_fitness(self, clusters):            

        if "arr" not in clusters[0]:
            output_clusters = []
            for cluster in clusters:
                entry = self.make_cluster_entry(cluster)
                output_clusters.append(entry)
            clusters = output_clusters

        if self.toprint:
            print("scoring")
        

        '''CH-Index'''
        
        labels = np.full(self.items, -1)
        for cluster_idx, cluster in enumerate(clusters):
            for point_id in cluster["arr"]:
                if point_id != -1:
                    labels[point_id] = cluster_idx
        if len(labels) > 1:
            ch_score = calinski_harabasz_score(
                self.cid_matrix[:self.items], 
                labels)
        else:
            ch_score = 0

        '''aisle separation'''
        if "al_sep" in self.metrics:
            pca = PCA(n_components=min(len(clusters), 4))
            means, cluster_aisles = self.loader.get_aisles(labels, self.cid_matrix_indices)

            normalized_aisles = []
            for i in range(len(cluster_aisles)):
                normalized_aisles.append(cluster_aisles[i] / cluster_aisles[i].sum())

            cluster_matrix = np.vstack(normalized_aisles)
            cluster_profiles = normalize(cluster_matrix, norm='l1', axis=1)
            normalized_aisles = pca.fit_transform(cluster_profiles)
            dist_matrix = euclidean_distances(cluster_profiles, cluster_profiles)
            mask = np.triu(np.ones_like(dist_matrix, dtype=bool), k=1)

            avg_separation = np.mean(dist_matrix[mask])
        else:
            avg_separation = 0
    
        '''silhouette'''
        if "sil" in self.metrics:
            sil = silhouette_score(self.cid_matrix, labels)
        else:
            sil = 0


        if self.toprint:
            print("entropy")
        
        def compute_entropy(matrix, clusters):
            entropies = []
            for i in range(len(clusters)):
                cids = clusters[i]["arr"]
                ids = matrix[cids, :]
                dist = np.asarray(np.sum(ids, axis=0)).flatten()
                dist += 0.01
                dist = dist / dist.sum()
                entropies.append(-np.sum(dist * np.log2(dist)) / np.log2(len(dist)))
            return entropies
        
        al_ent = compute_entropy(self.aid_matrix, clusters)
        dp_ent = compute_entropy(self.did_matrix, clusters)

        #between centroid separation
        bcs = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = np.linalg.norm(clusters[i]["mean"] - clusters[j]["mean"])
                bcs.append(dist)

        
        entry = {"clusters": clusters.copy(),
                "WCSS": -np.sum([cluster["ssd"] for cluster in clusters]),
                "CH-I": ch_score,
                "al_ent": np.var(al_ent),
                "al_sep": avg_separation,
                "dp_ent": np.var(dp_ent),
                "bcs": np.sum(bcs),
                "sil": sil,
                "b_size": 0,
                "cycle": self.cycle}
        if self.toprint:
            df = pd.DataFrame([entry])
            df = df.drop("clusters", axis=1)
            print(df.head())

        self.m1[tuple(tuple(cluster["arr"]) for cluster in clusters)] = entry


    def compute_cluster_fitness_old(self, clusters):            

        if "arr" not in clusters[0]:
            output_clusters = []
            for cluster in clusters:
                entry = self.make_cluster_entry(cluster)
                output_clusters.append(entry)
            clusters = output_clusters

        if self.toprint:
            print("silhouette")
        #compute silhouette (find closest clusters first to reduce computations)
        top_clusters = min(len(clusters), 3)
        child_dists = {k: {} for k in range(len(clusters))}
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = np.linalg.norm(clusters[i]["mean"] - clusters[j]["mean"])
                child_dists[i][j] = d
                child_dists[j][i] = d
        for i in range(len(clusters)):
            child_dists[i] = [item[0] for item in sorted(child_dists[i].items(), key=lambda x: x[1])[:top_clusters]]
        
        if self.toprint:
            print("scoring")
        #the score
        sils = []
        for i in range(len(clusters)):
            if self.toprint:
                print("    ", i)
            clust = clusters[i]["arr"]
            if len(clust) > 1:
                l = len(clust)
                pts = self.cid_matrix[clust, :]
                for j in range(l):
                    pt = pts[j, :]
                    ds = np.linalg.norm(pts - pt, axis=1)
                    a = np.sum(ds) / (l - 1)
                    mean_dists = []
                    for k in child_dists[i]:
                        pts_k = self.cid_matrix[clusters[k]["arr"], :]
                        pt_dists = np.linalg.norm(pts_k - pt, axis=1)
                        mean_dists.append(np.mean(pt_dists))
                    b = np.min(mean_dists) if len(mean_dists) > 0 else 0
                    sils.append((b - a) / max(a, b))


        if self.toprint:
            print("entropy")
        #computing per-cluster aisle entropy
        entropies = []
        for i in range(len(clusters)):
            cids = clusters[i]["arr"]
            alids = self.aid_matrix[cids, :]
            al_dist = np.asarray(np.sum(alids, axis=0)).flatten()
            al_dist += 0.01
            al_dist = al_dist / al_dist.sum()
            entropies.append(-np.sum(al_dist * np.log2(al_dist)) / np.log2(len(al_dist)))
        
        entry = {"clusters": clusters.copy(),
                "WCSS": -np.sum([cluster["ssd"] for cluster in clusters]),
                "sil": np.mean(sils),
                "al_ent": -np.mean(entropies),
                "cycle": self.cycle}
        if self.toprint:
            df = pd.DataFrame([entry])
            df = df.drop("clusters", axis=1)
            print(df.head())

        self.m1[tuple(tuple(cluster["arr"]) for cluster in clusters)] = entry




    def make_cluster_entry(self, child):

        child = np.array(child, dtype=np.int64)
        indices = child[child != -1]
        if len(indices) > 0:
            mean = np.mean(self.cid_matrix[indices, :], axis=0)
            ssd = np.sum((self.cid_matrix[indices, :] - mean) ** 2)

            entry = {"arr": child, "mean": mean, "ssd": ssd}
            return entry
        else:
            return {"arr": child, "mean": 0.0, "ssd": 0.0}
    

    def cross_clusters(self, survivors, n_workers=1):

        def recomp(cluster, idx, key):
            pt = self.cid_matrix[key, :]
            c = np.sum(cluster["arr"] != -1)
            new_mean = (cluster["mean"] * c + pt) / (c + 1)
            new_ssd = cluster["ssd"] + np.sum((cluster["mean"] - pt) ** 2) * (c / (c + 1))
            cluster.update({"mean": new_mean, "ssd": new_ssd})
            cluster["arr"][idx] = key
        
        def swap_pts(cluster_i, cluster_j, pt_i_idx, pt_j_idx):

            def swap(cluster, pt_a, pt_b):
                c = len(cluster["arr"])
                if c > 1:
                    x = cluster["ssd"] - np.sum((pt_a - cluster["mean"]) ** 2) * (c / (c - 1))
                    y = (cluster["mean"] * c - pt_a) / (c - 1)
                    x = x + np.sum((pt_b - y) ** 2) * ((c - 1) / c)
                    y = (y * (c - 1) + pt_b) / c
                    cluster["ssd"] = x
                    cluster["mean"] = y
                else:
                    cluster["ssd"] = 0
                    cluster["mean"] = pt_b

            pt_i = self.cid_matrix[cluster_i["arr"][pt_i_idx], :]
            pt_j = self.cid_matrix[cluster_j["arr"][pt_j_idx], :]

            swap(cluster_i, pt_i, pt_j)
            swap(cluster_j, pt_j, pt_i)

        
        def cleanup_keys_old(free_keys, clusters):
                if self.uniform:
                    n = np.random.choice(range(1, len(free_keys) + 1))
                elif len(free_keys) > self.k_range:
                    n = np.min([np.random.choice(range(1, len(free_keys) + 1)), np.random.poisson(lam=(len(free_keys) // self.k_range))])
                else:
                    n = len(free_keys)
                if self.toprint:
                    print("appending cluster", n)
                to_cut = min(n, len(free_keys))
                cluster = random.sample(list(free_keys), to_cut)
                free_keys -= set(cluster)
                entry = self.make_cluster_entry(cluster)
                clusters.append(entry)


        def cross_parents(p1, p2):

            if self.toprint:
                print("Cross-mutating", p1, p1)

            p1 = self.m1[survivors[p1]]
            p2 = self.m1[survivors[p2]]
            clust_list = [p1, p2]

            if len(p1["clusters"]) != len(p2["clusters"]):
                ref = clust_list[np.argmin([len(p1["clusters"]), len(p2["clusters"])])]
                tar = clust_list[np.argmax([len(p1["clusters"]), len(p2["clusters"])])]
            else:
                ref = p1
                tar = p2
            
            dists = {k: {} for k in range(len(ref["clusters"]))}
            for i in range(len(ref["clusters"])):
                for j in range(len(tar["clusters"])):
                    dists[i][j] = np.linalg.norm(ref["clusters"][i]["mean"] - tar["clusters"][j]["mean"])

            # for i in range(len(dists)):
            #     print(i, ref["clusters"][i]["mean"], tar["clusters"][sorted(dists[i].items(), key = lambda x: x[1])[0][0]]["mean"])
            

            n_children = np.random.poisson(lam=2)
            for child in range(n_children):
                dists_k = {key: dists[key].copy() for key in dists} #solved linked dicts issue
                free_keys = set()
                used_keys = set()
                dist_keys = list(dists_k.keys())
                clusters = []
                for i in range(len(dist_keys)):
                    if self.toprint:
                        print("crossing", i)
                    top = sorted(dists_k[dist_keys[i]].items(), key = lambda x: x[1])
                    closest = top[0]

                    # for j in range(len(dists_k)):
                    #     dists_k[dist_keys[j]].pop(closest[0]) #eliminating same-parent crosses hurt scores by a lot

                    #extremely similar logic to rule mining, except no duplicates and every cid must be accounted for
                    p = set(ref["clusters"][i]["arr"]) - used_keys
                    q = set(tar["clusters"][closest[0]]["arr"]) - used_keys

                    unique = p | q
                    r = p & q
                    u = list((p | q) - r)
                    # l = np.random.choice([len(p), len(q), len(p) - 1, len(p) + 1, len(q) - 1, len(q) + 1])
                    l = len(r) + min(len(p - r), len(q - r))
                    # l = len(p)

                    if l > len(r):
                        child = [np.int64(k) for k in r]
                        remainder = l - len(child)

                        if remainder < len(u) and remainder > 0:
                            child.extend(np.random.choice(u, remainder, replace=False))
                        else:
                            child.extend(u)
                            mutant = [-1 for _ in range(l - len(child))]
                            child.extend(mutant)
                    # elif l <= len(u):
                    #     child = list(np.random.choice(u, l, replace=False))
                    else: #tricky logic here, need to revisit this
                        child = list(u)
                        mutant = [-1 for _ in range(l - len(u))]
                        child.extend(mutant)
                    
                    child_set = set(item for item in child if item != -1)
                    used_keys.update(child_set)
                    free_keys.update(set(unique))

                    entry = self.make_cluster_entry(child)
                    clusters.append(entry.copy())

                if self.toprint:
                    print("cleanup")

                free_keys -= used_keys
                
                overflow_tuples = []
                for i in range(len(clusters)):
                    for j in range(len(clusters[i]["arr"])):
                        if clusters[i]["arr"][j] == -1:
                            overflow_tuples.append((i, j))

                free_keys = list(free_keys)
                if len(free_keys) > 0:
                    i = len(free_keys)
                    while len(overflow_tuples) > 0 and i > 0:
                        crd = overflow_tuples.pop(-1)
                        recomp(clusters[crd[0]], crd[1], free_keys[-1])
                        free_keys.pop(-1)
                        i -= 1

                if len(overflow_tuples) > 0:
                    for i in range(len(overflow_tuples) - 1, -1, -1):
                        crd = overflow_tuples[i]
                        clusters[crd[0]]["arr"] = np.delete(clusters[crd[0]]["arr"], crd[1])

                
                def cleanup_keys(free_keys, clusters):
                    '''assign remaining keys to the closest centroids by batch updating'''
                    cluster_dict = {i: {"arr": list(clusters[i]["arr"]),
                                        "mean": clusters[i]["mean"],
                                        "ssd": clusters[i]["ssd"]} for i in range(len(clusters))}
                    for key in free_keys:
                        min_dist = np.inf
                        min_cluster = -1
                        min_pt = 0
                        min_key = 0
                        pt = self.cid_matrix[key, :]
                        for j in range(len(cluster_dict)):
                            pt_dist = np.linalg.norm(pt - cluster_dict[j]["mean"])
                            if pt_dist < min_dist:
                                min_dist = pt_dist
                                min_cluster = j
                                min_pt = pt
                                min_key = key
                        c = len(cluster_dict[min_cluster]["arr"])
                        cluster_dict[min_cluster]["arr"].append(min_key)
                        cluster_dict[min_cluster]["ssd"] = cluster_dict[min_cluster]["ssd"] + np.sum((cluster_dict[min_cluster]["mean"] - min_pt) ** 2) * (c / (c + 1))
                        cluster_dict[min_cluster]["mean"] = (cluster_dict[min_cluster]["mean"] * c + min_pt) / (c + 1)

                    #update the actual clusters
                    for i in range(len(cluster_dict)):
                        clusters[i] = cluster_dict[i]
                
                def gen_centroids(n_clusts, clusters):
                    for i in range(n_clusts):
                        new_centroid = np.array([np.random.uniform(m, n) for m, n in self.cid_bounds])
                        clusters.append({"arr": np.array([]), "mean": new_centroid, "ssd": 0})
                

                k_diff = len(tar["clusters"]) - len(ref["clusters"])
                n_clusts = np.random.poisson(lam=k_diff)
                # gen_centroids(n_clusts, clusters)

                cleanup_keys(free_keys, clusters)

                total_length = sum([len(cluster["arr"]) for cluster in clusters])
                if total_length != self.cid_range:
                    dups = set()
                    for cluster in clusters:
                        dups.update(set(cluster["arr"]))
                    print("ERROR, LENGTH MISMATCH")
                    print(self.cid_range, total_length)
                    print(free_keys, overflow_tuples)
                    print(dups)
                    input()


                #handle random mutation here because it's easier
                index_lengths = {}
                cumulative = 0
                for i in range(len(clusters)):
                    start = cumulative
                    end = cumulative + len(clusters[i]["arr"])
                    index_lengths[i] = (start, end)
                    cumulative = end

                for i in range(round(self.cid_range * self.mut_rate)):
                    pt_i_idx = np.random.randint(0, self.cid_range)
                    pt_j_idx = np.random.randint(0, self.cid_range)
                    if self.toprint:
                        print("swapping", pt_i_idx, pt_j_idx)
                    while pt_i_idx == pt_j_idx:
                        pt_j_idx = np.random.randint(0, self.cid_range)
                    for k, (start, end) in index_lengths.items():
                        if start <= pt_i_idx < end:
                            cluster_i = clusters[k]
                            pt_i_idx -= start
                        if start <= pt_j_idx < end:
                            cluster_j = clusters[k]
                            pt_j_idx -= start

                    swap_pts(cluster_i, cluster_j, pt_i_idx, pt_j_idx)
                
                to_pop = []
                for i in range(len(clusters)):
                    if len(clusters[i]["arr"]) == 0:
                        to_pop.append(i)
                for i in range(len(to_pop) - 1, -1, -1):
                    clusters.pop(to_pop[i])

                
                self.compute_cluster_fitness(clusters)
                
        indices = [i for i in range(len(survivors))]
        pairs = []
        for i in range(len(survivors) // 2):
            pair = []
            for j in range(2):
                p = random.sample(indices, 1)[0]
                indices.remove(p)
                pair.append(p)
            pairs.append(tuple(pair))

        # for pair in pairs:
        #     cross_parents(pair[0], pair[1])
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(lambda pair: cross_parents(pair[0], pair[1]), pairs))

    
    def soft_cluster(self, survivors):
        from sklearn.decomposition import NMF

        coocm = np.full((self.cid_range, self.cid_range), 0)
        lengths = []
        for i in range(len(survivors)):
            clusters = survivors[i]
            lengths.append(len(clusters))
            for cluster in clusters:
                coocm[np.ix_(cluster, cluster)] += 1
        
        k = round(np.mean(lengths))
        coocm_norm = coocm / coocm.max()
        nmf = NMF(n_components=k, init='nndsvda', random_state=42, max_iter=500)
        W = nmf.fit_transform(coocm_norm)
        H = nmf.components_
        fuzzy_memberships = W / W.sum(axis=1, keepdims=True)
        return fuzzy_memberships

        




    def cross_mutate(self, survivors):

        if self.toprint:
            print("crossing parents")

        def choose_nonredundant_random(n, child):
            child_set = set(child)  # Convert once for faster lookup
            while True:
                if self.uniform:
                    mutant = np.random.choice(self.items, n, replace=False)
                else:
                    mutant = np.random.choice(self.items, n, p=self.pdist, replace=False)
                
                # Check if ANY element in mutant is already in child
                if not any(m in child_set for m in mutant):
                    return mutant
                
        def make_child(p, q, r, u):
            sd = abs(len(p) - len(q)) / 4 + 1.0
            mean = (len(p) + len(q)) / 2
            l = int(np.random.normal(mean, sd))
            l = max(2, l)

            # l = np.random.choice([len(p), len(q), len(p) - 1, len(p) + 1, len(q) - 1, len(q) + 1])
            # l = len(r) + min(len(p - r), len(q - r))
            if l > len(r):
                child = [np.int64(k) for k in r]
                remainder = l - len(child)

                if remainder < len(u) and remainder > 0:
                    child.extend(np.random.choice(u, remainder, replace=False))
                else:
                    child.extend(u)
                    mutant = choose_nonredundant_random(remainder, child)
                    child.extend(mutant)
            elif l <= len(r):
                child = list(np.random.choice(r, l, replace=False))
            return child


        def crossover(p, q):
            children = []
            p = set(p)
            q = set(q)
            r = (p & q)
            u = list((p | q) - r)
            n_children = np.random.poisson(lam=2)
            for i in range(n_children):
                child = make_child(p, q, list(r), u)
                pids, tids, count = self.get_itemset_support_basic(child)
                while np.sum(count) == 0:
                    child = make_child(p, q, list(r), u)
                    pids, tids, count = self.get_itemset_support_basic(child)
                children.append(child)
            return children

        random.shuffle(survivors)

        n = 0
        while n < 1:
            offspring = []
            for i in range(0, len(survivors) - 1, 2):
                if self.toprint:
                    print(i)
                p = survivors[i]
                q = survivors[i + 1]
                children = crossover(p, q)
                offspring.extend(children)
            n = len(offspring)

        if self.toprint:
            print("mutating the children")
        
        muts = round(n * self.mut_rate)
        to_mutate = np.random.choice(range(len(offspring)), muts, replace=True)
        for i in to_mutate:
            if self.toprint:
                print(i)
            x = random.randint(0, 1)
            if x == 0 and len(offspring[i]) > 2:
                offspring[i].remove(np.random.choice(offspring[i]))
            else:
                pids, tids, counts = self.get_itemset_support_basic(offspring[i])
                if np.sum(counts) > self.minsup:
                    tries = 0
                    mutant = choose_nonredundant_random(1, offspring[i])
                    test_child = list(offspring[i]) + list(mutant)
                    pids, tids, counts = self.get_itemset_support_basic(test_child)
                    while np.sum(counts) < self.minsup and tries < 100:
                        tries += 1
                        mutant = choose_nonredundant_random(1, offspring[i])
                        test_child = list(offspring[i]) + list(mutant)
                        pids, tids, counts = self.get_itemset_support_basic(test_child)
                    if tries <= 100 and np.sum(counts) > 0:
                        offspring[i].extend(mutant)
        sorted_children = [tuple(sorted(child)) for child in offspring]
        self.popn = survivors + sorted_children

    '''auxiliary code -------------------------------------------------------------'''
    def test_miner(self, i, k, j, toprint=False):
        self.toprint = toprint

        np.set_printoptions(formatter={'all': lambda x: str(x)}, legacy='1.13')
        while True:

            n = np.random.randint(1, j)
            ids = tuple(sorted(np.random.choice(self.high_item_ids[:k], n, replace=False)))

            self.compute_itemset_fitness(ids, i)
            if toprint:
                m2_list = [{"key": kp, "tids": self.m2[kp]["tids"], "counts": self.m2[kp]["counts"].sum()} for kp in self.m2]
                m2_df = pd.DataFrame(m2_list)
                print(m2_df)
                print(self.m1)
                input("ENTER to Continue")
    
    def get_support_mapping(self):
        if self.testmode:
            items = self.high_item_ids
            length = len(self.high_item_ids)
            map_type = "highitem"
        else:
            items = self.items
            length = len(items)
            map_type = "dense"
        if not os.path.exists(self.path / f"support_map_{map_type}.parquet"):
            supports = {}
            for i in range(length):
                print(i)
                product_ids, tids, counts = self.get_itemset_support_basic([items[i]])
                supports[product_ids[0]] = sum(counts) / self.N
            sort_by_sup = sorted(supports.items(), key = lambda x: x[1], reverse=True)
            to_df = []
            for i in range(len(sort_by_sup)):
                entry = sort_by_sup[i]
                to_df.append({"new_idx": i, "old_idx": entry[0]})
            df = pd.DataFrame(to_df)
            df.to_parquet(self.path / f"support_map_{map_type}.parquet")
        prod_map = pd.read_parquet(self.path / f"support_map_{map_type}.parquet")
        mapping = dict(zip(prod_map["old_idx"], prod_map["new_idx"]))
        return mapping


    def dot_plot_popn(self, order_sup=True):

        if order_sup:
            mapping = self.get_support_mapping()
                
        popn, cyc = self.load_popn(return_obj=False)

        #flatten popn
        cycles = []
        for i in range(len(popn)):
            if order_sup:
                array = np.array(list(chain.from_iterable(popn[i])))
                remap = np.array([mapping[val] for val in array if val in mapping])
                cycles.append(remap)
            else:
                cycles.append(np.array(list(chain.from_iterable(popn[i]))))


        fig, ax = plt.subplots(figsize=(12, 8))

        for i in range(len(popn)):
            ax.scatter(cycles[i], np.full(len(cycles[i]), i), alpha=.5, s=20)

        ax.set_xlabel('Value')
        ax.set_ylabel('Cycle')
        ax.set_title('Gene Map')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def kde_plot_popn(self, order_sup=True):

        from sklearn.neighbors import KernelDensity

        if order_sup:
            mapping = self.get_support_mapping()
                    
        popn, cyc = self.load_popn(return_obj=False)

        #flatten popn
        cycles = []
        for i in range(len(popn)):
            if order_sup:
                array = np.array(list(chain.from_iterable(popn[i])))
                remap = np.array([mapping[val] for val in array if val in mapping])
                cycles.extend(remap)
            else:
                cycles.extend(np.array(list(chain.from_iterable(popn[i]))))

        cycles = np.array(cycles)

        # Create uniform kernel KDE (tophat = uniform)
        kde = KernelDensity(kernel='linear', bandwidth=1.0)
        kde.fit(cycles.reshape(-1, 1))

        # Evaluate on grid
        x_range = np.linspace(cycles.min(), cycles.max(), 1000)
        log_density = kde.score_samples(x_range.reshape(-1, 1))
        density = np.exp(log_density)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x_range, density, linewidth=2)
        ax.fill_between(x_range, density, alpha=0.3)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Gene Distribution KDE')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    
    def iiid_plot_popn(self):
        import plotly.graph_objects as go

        self.load_memos(u=1)
        data = self.m1

        pts = []
        for key in data:
            pts.append(((np.log(data[key]["sup"])), data[key]["conf"], np.log(data[key]["lift"])))

        support, confidence, lift = zip(*pts)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        img = ax.scatter(support, confidence, lift, color="blue", cmap='PuBu', marker='o', alpha=0.9)

        ax.set_xlabel('Log(Support)')
        ax.set_ylabel('Confidence')
        ax.set_zlabel('Log(Lift)')
        ax.set_title('Association Rules: Support vs Confidence vs Lift')
        # fig.colorbar(img, ax=ax, label='Lift')

        plt.show()

    def write_rules_csv(self):

        self.load_memos(u=1)
        data = self.m1

        prod_map = pd.read_parquet(self.path / "hot_map_products.parquet")
        prod_names = pd.read_csv(self.path.parent / "instacart_data" / "products.csv")

        pts = []
        for key in data:
            ant_ids = [prod_map[prod_map["col_id"] == id]["product_id"].iloc[0] for id in data[key]["tag"][0]]
            coq_ids = [prod_map[prod_map["col_id"] == id]["product_id"].iloc[0] for id in data[key]["tag"][1]]
            ant_names = [prod_names[prod_names["product_id"] == pid]["product_name"].iloc[0] for pid in ant_ids]
            coq_names = [prod_names[prod_names["product_id"] == pid]["product_name"].iloc[0] for pid in coq_ids]

            pts.append({"antecedent": data[key]["tag"][0],
                        "consequent": data[key]["tag"][1],
                        "ant_name": ant_names,
                        "coq_name": coq_names,
                        "support": data[key]["sup"], 
                        "confidence": data[key]["conf"], 
                        "lift": data[key]["lift"],
                        "iteration": data[key]["cycle"]
                        })
        
        df = pd.DataFrame(pts)
        df.to_csv(self.memo_path / "NSGA_rules.csv", index=False)
    
    def write_clusters_parquet(self, to_max="WCSS"):
        import re

        fuzzy_matrix = pd.read_pickle(self.memo_path / "fuzzy_memberships.pkl")
        fuzzy_matrix = fuzzy_matrix.values + 1e-10
        fuzzy_matrix = fuzzy_matrix / fuzzy_matrix.sum(axis=1, keepdims=True)
        fuzzy_cluster_assignments = np.argmax(fuzzy_matrix, axis=1)

        data = {}
        with open(self.memo_path / "memo_1.jsonl") as f:
            for line in f:
                entry = json.loads(line)
                for str_key, value in entry.items():
                    key = tuple(int(x) for x in re.findall(r'\d+', str_key))
                    converted_value = convert_to_numpy(value)
                    data[key] = converted_value

        data_sorted = sorted(data.items(), key = lambda x: x[1][to_max], reverse=True)
        best = data_sorted[:10]

        results = []

        for entry in best:
            value = entry[1]  # entry[0] is the key, entry[1] is the value
            labels = np.zeros(len(fuzzy_cluster_assignments), dtype=int)
            for i in range(len(value["clusters"])):
                for j in range(len(value["clusters"][i]["arr"])):
                    labels[value["clusters"][i]["arr"][j]] = int(i)

            results.append({"type": "best", "labels": labels, "tsvd": 8, "k": len(value["clusters"]), "WCSS": -value["WCSS"]})

        fuz_wcss = 0
        for i in range(len(np.unique(fuzzy_cluster_assignments))):
            pts = self.cid_matrix[fuzzy_cluster_assignments == i]
            if len(pts) > 0:
                centroid = pts.mean(axis=0)
                fuz_wcss += np.sum((pts - centroid) ** 2)

        results.append({"type": "fuzzy", "labels": fuzzy_cluster_assignments, "tsvd": 8, "k": fuzzy_matrix.shape[1], "WCSS": fuz_wcss})

        df = pd.DataFrame(results)
        df.to_parquet(self.path / "NSGA_clusters.parquet")





    def iiid_plot_multi(self, filenames=["minsup 20, mut 0.1, 30 trials, sup jacc, VL/memo_1.jsonl", "minsup 20, mut 0.1, 30 trials, sup conf, VL/memo_1.jsonl"]):

        colors = ["blue", "purple", "green", "red", "orange"] # Added a few more just in case
        names = ["Support, Jaccard","Support, Confidence"]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')


        for i, file in enumerate(filenames):
            if True:
                # 1. Generate a Label: Clean up the filename to make it readable in the legend
                # This splits by '/' and takes the first part (the folder name usually)
                label_name = names[i]
                
                data = {}
                # Ensure self.memo_path is a Path object. If it's a string, change to: Path(self.memo_path) / file
                full_path = self.memo_path / file 
                
                with open(full_path, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        for str_key, value in entry.items():
                            # Parsing logic maintained from your snippet
                            key = tuple(np.int64(x) for x in ast.literal_eval(str_key))
                            # Assuming convert_to_numpy is defined elsewhere in your class/global scope
                            converted_value = convert_to_numpy(value) # Added 'self.' if it's a method
                            data[key] = converted_value

                pts = []
                for key in data:
                    # Safety check for log(0)
                    sup = data[key]["sup"]
                    lift_val = data[key]["lift"]
                    
                    # Avoid errors if support or lift are 0
                    log_sup = np.log(sup) if sup > 0 else 0
                    log_lift = np.log(lift_val) if lift_val > 0 else 0
                    
                    pts.append((log_sup, data[key]["conf"], log_lift))

                if pts:
                    support, confidence, lift = zip(*pts)
                    if i == 0:
                        op = 1
                    else:
                        op = 0.5
                    # 2. Add the 'label' argument here
                    ax.scatter(support, confidence, lift, 
                            color=colors[i % len(colors)], 
                            marker='o', 
                            alpha=op, 
                            label=label_name)

        ax.set_xlabel('Log(Support)')
        ax.set_ylabel('Confidence')
        ax.set_zlabel('Log(Lift)')
        ax.set_title('Association Rules: Support vs Confidence vs Lift')
        
        # 3. Use Legend instead of Colorbar
        ax.legend(loc='best')
        ax.legend(title="Objectives", loc='best')

        plt.show()



    
    def plot_fuzzy_umap(self):
        matrix = pd.read_pickle(self.memo_path / "fuzzy_memberships.pkl")
        matrix = matrix.values + 1e-10
        matrix = matrix / matrix.sum(axis=1, keepdims=True)

        from umap import UMAP

        c_list = [
            (1.0, 0.0, 0.0),      # 0: Red
            (0.0, 1.0, 0.0),      # 1: Green
            (0.0, 0.0, 1.0),      # 2: Blue
            (1.0, 1.0, 0.0),      # 3: Yellow
            (1.0, 0.0, 1.0),      # 4: Magenta
            (0.0, 1.0, 1.0),      # 5: Cyan
            (1.0, 0.5, 0.0),      # 6: Orange
            (0.5, 0.0, 0.5),      # 7: Purple
            (0.0, 0.5, 0.0),      # 8: Dark Green
            (0.5, 0.5, 0.0),      # 9: Olive
            (0.0, 0.5, 0.5),      # 10: Teal
            (0.5, 0.0, 0.0),      # 11: Maroon
            (1.0, 0.75, 0.8),     # 12: Pink
            (0.6, 0.4, 0.2),      # 13: Brown
            (0.5, 0.5, 0.5),      # 14: Gray
            (1.0, 0.84, 0.0),     # 15: Gold
            (0.5, 1.0, 0.0),      # 16: Lime
            (1.0, 0.0, 0.5),      # 17: Hot Pink
            (0.0, 1.0, 0.5),      # 18: Spring Green
            (0.5, 0.0, 1.0),      # 19: Blue-Violet
        ][:matrix.shape[1]]

        umapper = UMAP()
        pts = self.cid_matrix[:self.cid_range, :]
        pts_2d = umapper.fit_transform(pts)

        mixed_colors = []
        for i in range(matrix.shape[0]):
            responsibility = matrix[i, :]
            mixed_color = np.sum([responsibility[j] * np.array(c_list[j]) 
                        for j in range(len(c_list))], axis=0)
            mixed_colors.append(mixed_color)

        plt.figure(figsize=(10, 8))
        plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=mixed_colors, s=50, alpha=0.6)
        plt.title("Fuzzy Cluster Visualization (UMAP)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        
        # Add legend showing pure cluster colors
        for i, color in enumerate(c_list):
            plt.scatter([], [], c=[color], label=f'Cluster {i}', s=100)
        plt.legend()
        plt.show()

        








    def genetically_modify(self, n_workers=1, cycles=10):
        self.load_memos()
        self.load_popn()

        if self.cycle == 1 or len(self.m1) == 0:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(lambda p: self.compute_itemset_fitness(p, self.combi_limit), self.popn))
        
        for i in range(cycles):

            survivors = self.select_survivors()
            self.cross_mutate(survivors)
            print("SURVIVORS", len(survivors))

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(lambda p: self.compute_itemset_fitness(p, self.combi_limit), self.popn))

            self.prune_memos()
            self.cycle += 1
            self.save_popn()
        self.save_memos()
            
        self.dot_plot_popn()
        self.kde_plot_popn()

    
    def genetically_cluster(self, n_workers = 1, cycles=10):
        self.load_memos()
        self.init_clusters()
        # self.load_popn()
        self.cycle = 1


        if self.cycle == 1 or len(self.m1) == 0:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(lambda p: self.compute_cluster_fitness(p), self.popn))
        
        for i in range(cycles):

            survivors = self.select_survivors()
            print("SURVIVORS", len(survivors))
            self.cross_clusters(survivors, n_workers=n_workers)

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(lambda p: self.compute_cluster_fitness(p), self.popn))

            self.prune_memos()
            self.cycle += 1
            if i % 10 == 0:
                self.save_popn()
        self.save_memos()
    
        survivors = self.select_survivors(return_fittest=True)
        fuzzy_clusters = self.soft_cluster(survivors)
        fuzzy_df = pd.DataFrame(fuzzy_clusters)
        fuzzy_df.to_pickle(self.memo_path / 'fuzzy_memberships.pkl')
        self.plot_fuzzy_umap()


        keys = list(self.m1.keys())
        for_df = []
        for key in keys:
            entry = {metric: self.m1[key][metric] for metric in self.metrics}
            if "WCSS" not in entry:
                entry["WCSS"] = self.m1[key]["WCSS"]
            if "CH-I" not in entry:
                entry["CH-I"] = self.m1[key]["CH-I"]
            entry["k"] = len(self.m1[key]["clusters"])
            for_df.append(entry)
        for_df = sorted(for_df, key = lambda x: -x["WCSS"])
        df = pd.DataFrame(for_df)
        df.to_csv(self.memo_path / "df.csv")

        # from sklearn.cluster import KMeans
        # clu = KMeans(n_clusters=3,algorithm="elkan")
        # clu.fit_predict(self.cid_matrix[:self.cid_range])
        # chi = calinski_harabasz_score(self.cid_matrix[:self.cid_range], clu.labels_)
        # print("KMeans wcss", clu.inertia_)
        # print("CH-Index", chi)





# geno = Gen_Optimizer(smoothing_alpha=0, uniform=True, mode="rulemine", 
#                      pop_cap=1000, mut_rate=0.1, k_range=(2, 3), 
#                      toprint=True, metrics=["sup", "conf"])
# geno.genetically_modify(cycles=60)
# geno.dot_plot_popn()
# geno.iiid_plot_popn()
# geno.iiid_plot_multi(["minsup 20, mut 0.1, 30 trials, sup conf, VL/memo_1.jsonl", "memo_1.jsonl"])
# geno.write_rules_csv()


genc = Gen_Optimizer(mode = "clustomers", uniform=True, pop_cap=100, toprint=True, mut_rate=0.005, metrics=["al_sep", "sil"])
# genc.genetically_cluster(n_workers=12, cycles=50)
genc.write_clusters_parquet(to_max = "CH-I")



'''notes: init_popn is set to self.pdist by default since complete random sampling takes forever with possible combinations
added checks to miner mutations and crossover functions to prevent
Need to regenerate tidset_dense if changing minsup count'''