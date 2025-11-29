import pandas as pd
import numpy as np
from pathlib import Path
import platform
import os

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import spmatrix

from itertools import combinations
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
        # Check if it's a numeric list that should be a numpy array
        if obj and isinstance(obj[0], (int, float)):
            return np.array(obj)
        return [convert_to_numpy(item) for item in obj]
    elif isinstance(obj, int):
        return np.int64(obj)
    elif isinstance(obj, float):
        return np.float64(obj)
    return obj

class Gen_Optimizer:

    def __init__(self, filename, sys_username="Eric Arnold", smoothing_alpha=1, cycle_limit=10, combi_limit=3, pop_cap=1000):
        self.cycle_limit = cycle_limit
        self.combi_limit = combi_limit
        self.username = sys_username
        self.pop_cap = pop_cap

        self.high_item_ids = [24849, 13173, 21134, 21900, 47205, 47762, 47622, 16794, 26206, 27842, 27963, 22932, 24961, 45004, 39272, 49679, 28201, 5874, 40703, 8274, 4918, 30388, 45063, 42262, 44629, 49231, 19054, 4603, 21613, 37643, 17791, 27101, 30486, 31714, 27083, 46975, 8515, 44356, 28982, 41947, 26601, 5075, 34123, 22032, 39874, 43349, 35948, 10746, 19657, 9073, 24181, 21935, 43958, 34966, 48675, 46663, 12338, 25887, 31503, 5448, 39925, 24835, 22822, 5783, 35218, 28839, 33728, 8421, 27518, 33195, 8171, 44139, 20111, 27341, 11517, 29484, 18462, 28196, 15287, 46902, 9836, 27153, 3955, 43119, 23906, 34355, 4797, 9384, 16756, 195, 42733, 4208, 38686, 41784, 47140, 41217, 7778, 32997, 20992, 21706]

        self.m1 = {}
        self.m2 = {}
        self.memo_filenames = ["memo_1.jsonl", "memo_2.jsonl"]

        if platform.system() == "Windows":
            self.memo_path = Path("C:/") / "Users" / self.username / "Documents" / "NSGA_data"
        elif platform.system() == "Darwin":
            self.memo_path = Path.home() / "Documents" / "NSGA_data"

        if not self.memo_path.exists():
            self.memo_path.mkdir(parents=True, exist_ok=True)

        self.loader = CSR_Loader()
        self.filename = filename
        self.path = Path(__file__).parent / "parquet_files"
        dense_name = f"{filename}_tidset_dense.parquet"

        if not os.path.isfile(self.path / dense_name):
            print("initializing matrices")

            matrix = self.loader.load(filename)
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
        tidset_sparse = self.loader.load(filename)
        self.N = tidset_sparse.sum()
        self.tidset_sparse = tidset_sparse.tocsc()
        self.pid_map = pd.read_parquet(self.path / "hot_map_products.parquet")
        pdist = np.array(tidset_sparse.sum(axis=0)).flatten() / self.N
        pdist += smoothing_alpha
        self.pdist = pdist / pdist.sum()
        self.item_id_range = tidset_sparse.shape[1]


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

        min_prod = iset_df["len"].idxmin()
        tids = iset_df.loc[min_prod, "tids"]

        sparse_rows = self.tidset_sparse[:, product_ids]
        sparse_filtered = sparse_rows[tids, :]
        binary_intersection = sparse_filtered.sign()
        cap = np.where(binary_intersection.getnnz(axis=1) == binary_intersection.shape[1])[0]
        if len(cap) > 0:
            # counts = sparse_filtered[cap, :].sum(axis=1).A1
            counts = sparse_filtered[cap, :].min(axis=1).A
            filtered_tids = tids[cap]
            return product_ids, np.array(filtered_tids), counts
        else:
            return product_ids, np.array([]), np.array([])

    def update_memo_2(self, key, tids, counts):
        self.m2[key] = {"tids": tids, "counts": counts, "sum": counts.sum(), "cycle": self.cycle}

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

        if len(pfx_key) > 0:
            pfx = self.m2[pfx_key]
            iset_df = self.tidset_dense.loc[product_ids, ["tids", "len"]]
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
                counts = np.minimum(sparse_filtered[cap, :].min(axis=1).A, pfx_counts[cap])
                filtered_tids = tids[cap]
                self.update_memo_2(key_all, np.array(filtered_tids), np.array(counts))
            else:
                self.update_memo_2(key_all, np.array([]), np.array([]))
        else:
            product_ids, tids, counts = self.get_itemset_support_basic(product_ids)
            self.update_memo_2(product_ids, tids, counts)


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
                ants[ant_key] = self.m2[ant_key]["sum"] / self.N
        sup_all = self.m2[product_ids]["sum"] / self.N
        ant_key, sup_ant = min(ants.items(), key=lambda x: x[1]) #select the lowest_support antecedent to get highest-confidence rule
        coq_key = tuple(x for x in product_ids if x not in set(ant_key))
        if coq_key not in self.m2:
            self.get_itemset_support(tuple(), coq_key)
        sup_coq = self.m2[coq_key]["sum"] / self.N

        if sup_all != 0:
            conf = sup_all / sup_ant
            lift = sup_all / (sup_coq * sup_ant)
        else:
            conf = 0
            lift = 0

        self.m1[product_ids] = {"tag": (ant_key, coq_key),
                                "sup": sup_all,
                                "conf": conf, 
                                "lift": lift, 
                                "length": len(product_ids),
                                "cycle": self.cycle
                                }


    def compute_fitness(self, ivec, i):

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
                cap_counts = np.minimum(t1["counts"][np.searchsorted(tids1, cap_tids)],
                                        t2["counts"][np.searchsorted(tids2, cap_tids)])
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
            return self.m1[key]
        
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
            print("ORIG", original_key)
            print("P", prefix, "K", key)

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
                    entry = {str(tuple(int(x) for x in key)): value}
                    f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")
    
    def prune_memos(self):

        memos = [self.m1, self.m2]
        for i in range(2):
            to_pop = []
            for key in memos[i]:
                if memos[i][key]["cycles"] < self.cycles - self.cycle_limit:
                    to_pop.append(key)

            for key in to_pop:
                memos[i].pop(key)

    def load_popn(self):

        filepath = self.memo_path / "popn.jsonl"

        if os.path.exists(filepath):
            print("load popn from saved")
            with open(self.memo_path / "popn.jsonl") as f:
                line = deque(f, maxlen=1)[0]
                line = json.loads(line)
                self.popn = [tuple(individual) for individual in line["population_array"]]
                self.cycle = line["cycle"]
                print("loaded cycle", self.cycle)
        else:
            print("genereated popn")
            self.popn = self.init_popn(test=True)
            self.cycle = 1
    
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
    def init_popn(self, k_range=(2, 10), uniform=False, test=False):
        if test:
            items = self.high_item_ids
            uniform = True
        else:
            items = self.item_id_range

        population = []
        for _ in range(self.pop_cap):
            k = np.random.choice(k_range)
            if uniform:
                ind = tuple(sorted(np.random.choice(items, k, replace=False)))
            else:
                ind = tuple(sorted(np.random.choice(items, k, replace=False, p=self.pdist)))
            population.append(ind)
        return population
    
    # def sort_individuals(self):

    #     metrics = ["sup", "conf", "lift"]
        
    #     arrays = [[] for _ in range(len(metrics))]

    #     for i in range(len(metrics)):
    #         arrays[i] = sorted({k: self.m1[k][metrics[i]] for k in self.m1}.items(), key = lambda x: x[1], reverse=True)

    def select_survivors(self):

        '''find fronts, comput crowding distance for last front up to length self.pop_cap'''

        metrics = ["sup", "conf", "lift"]
        m = len(metrics)

        def is_dom(p, q):
            if sum([int(self.m1[p][metrics[x]] > self.m1[q][metrics[x]]) for x in range(m)]) == m:
                dom_keys[p].add(q)
                dom_counts[q] += 1
            elif sum([int(self.m1[q][metrics[x]] > self.m1[p][metrics[x]]) for x in range(m)]) == m:
                dom_keys[q].add(p)
                dom_counts[p] += 1
        
        def crowding_distance(k, front):



                
        
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


        return sorted_individuals


    def cross_mutate(self):
        pass








    def genetically_modify(self, n_workers=2, cycles=10):
        self.toprint=True
        self.load_memos(u = 1)
        self.load_popn()
        
        for i in range(cycles):
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                list(executor.map(lambda p: self.compute_fitness(p, self.combi_limit), self.popn))
            self.cycle += 1
            survivors = self.select_survivors()
            self.cross_mutate(survivors)
            self.save_memos()
            self.save_popn()



    

    def test_miner(self, i, k, j, toprint=False):
        self.toprint = toprint

        np.set_printoptions(formatter={'all': lambda x: str(x)}, legacy='1.13')
        while True:

            n = np.random.randint(1, j)
            ids = tuple(sorted(np.random.choice(self.high_item_ids[:k], n, replace=False)))

            self.compute_fitness(ids, i)
            if toprint:
                m2_list = [{"key": kp, "tids": self.m2[kp]["tids"], "counts": self.m2[kp]["counts"].sum()} for kp in self.m2]
                m2_df = pd.DataFrame(m2_list)
                print(m2_df)
                print(self.m1)
                input("ENTER to Continue")




geno = Gen_Optimizer("hot_customers_products")
# geno.test(3, 10, 5, toprint=True)
geno.genetically_modify(cycles=1)