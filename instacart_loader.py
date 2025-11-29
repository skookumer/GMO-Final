import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import os

import pyarrow as pa
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, coo_matrix


path = Path(__file__).parent / "instacart_data"
write_path = Path(__file__).parent / "parquet_files"

def write_csr_as_coo(data, name):
    if not os.path.isfile(write_path / name):
        coo = data.tocoo()

        sparse_table = pd.DataFrame({
            "row_id": coo.row, 
            "col_id": coo.col, 
            "value": coo.data
        })

        table = pa.Table.from_pandas(sparse_table, preserve_index=False)
        pq.write_table(table, write_path / name)
        print(f"written {name}")
    else:
        print(f"skipped {name}")

def get_mappings(column_name, target):
    cat = target[column_name].astype("category")
    idx = cat.cat.codes
    unique = cat.cat.categories
    return {"cat": cat, "idx": idx, "unique": unique}

def write_map(map, name, filename, column_name):
    df = pd.Series(map["unique"], name=f'{name}_id').reset_index().rename(columns={'index': column_name})
    if not os.path.isfile(write_path / filename):
        df.to_parquet(write_path / filename)
        print(f"written {filename}")
    else:
        print(f"skipped {filename}")

def calc_csr(reference, map_a, map_b):
    dim = (len(map_a["unique"]), len(map_b["unique"]))
    sparse_coo = coo_matrix((np.ones(len(reference)), 
                                (map_a["idx"], map_b["idx"])), 
                                shape=dim)
    return sparse_coo.tocsr()

def read_instacart_old():
    path = Path(__file__).parent / "instacart_data" 
    users = pd.read_csv(path / "orders.csv")
    orders = pd.read_csv(path / "order_products__prior.csv")
    orders2 = pd.read_csv(path / "order_products__train.csv")
    orders = pd.concat([orders, orders2], ignore_index=True)
    products_df = pd.read_csv(path / "products.csv")

    transaction_database = defaultdict(dict)
    aligned_keys = []
    max_user = np.max(users["user_id"])
    max_product = 49689
    k = 0
    for i in range(1, max_user + 1):
        print(i)
        order_list = []
        row = dict(users.iloc[k])
        while k < len(users): # Check boundary first
            row = dict(users.iloc[k]) 
            if row["user_id"] == i:
                order_list.append(row["order_id"])
                k += 1 # Move to the next row for the NEXT iteration
            else:
                # We hit the next user, stop processing orders for user i
                break
        tid_vector = []
        for order in order_list:
            products = list(orders[orders["order_id"] == order]["product_id"])
            for product in products:
                if product not in aligned_keys:
                    aligned_keys.append(product)
                tid_vector.append(aligned_keys.index(product))
        tv_dict = {}
        for val in tid_vector:
            # item = int(products_df.loc[val, "product_id"])
            item = aligned_keys[val]
            if item in tv_dict:
                tv_dict[item] += 1
            else:
                tv_dict[item] = 1
        print(tv_dict)
        input()
        if len(tid_vector) < max_product:
            tid_vector += [0 for o in range(max_product - len(tid_vector))]
        transaction_database[i] = tid_vector
    
    transactions = np.array(transaction_database)
    print(transactions)
    print(transactions.size)


def process_parquet():

    files = [
    "hot_baskets_products.parquet", "hot_map_products.parquet", "hot_map_orders.parquet",
    "hot_customers_products.parquet", "hot_map_custs.parquet",
    "hot_map_groceries_products.parquet", "hot_map_groceries_customers.parquet"
    ]
    go = False
    for file in files:
        if os.path.isfile(write_path / file) is False:
            go = True

    if go:
        users = pd.read_csv(path / "orders.csv")
        print("read users")
        orders = pd.read_csv(path / "order_products__prior.csv")
        orders2 = pd.read_csv(path / "order_products__train.csv")
        orders = pd.concat([orders, orders2], ignore_index=True)
        print("read orders")
        products = pd.read_csv(path / "products.csv")
        print("read products")

        merged = users.merge(orders, on="order_id", how="inner")
        print("merged frames")
        # aisle_counts = merged.groupby(["user_id", "aisle_id"]).size().reset_index(name="count")
        # for index, row in merged.iterrows():
        #     print(f"Index: {index}, Order ID: {row['order_id']}, User ID: {row['user_id']}")
        #     input()
        # print(aisle_counts.head())
        # bb = "basic_baskets.parquet"
        # if not os.path.isfile(write_path / bb):
        #     bbdf = merged.groupby(['user_id', 'order_id']).agg(product_ids=('product_id', list)).reset_index()
        #     print("aggregated baskets")
        #     bbdf.to_parquet(write_path / bb)
        #     print(f"{bb} written")
        # else:
        #     print(f"{bb} exists")

        '''WRITE ORDER MATRIX AS prod_id, dept_id, aisle_id'''

        print("Generating sparse vectors and maps...")
        
        order_map = get_mappings("order_id", merged)
        product_map = get_mappings("product_id", merged)
        
        bbv_csr = calc_csr(merged, order_map, product_map)

        write_map(product_map, "product", "hot_map_products.parquet", "col_id")
        write_map(order_map, "order", "hot_map_orders.parquet", "row_id")
        
        write_csr_as_coo(bbv_csr, "hot_baskets_products.parquet")

        # Alternative mappings
        product_map_df = pd.Series(product_map["unique"], name='product_id').reset_index().rename(columns={'index': 'col_id'})
        attribute_map = products[['product_id', 'aisle_id', 'department_id']].copy()
        attribute_map = attribute_map.merge(
            product_map_df[['product_id', 'col_id']], 
            on='product_id', 
            how='inner'
        )
        
        aisle_map = get_mappings("aisle_id", attribute_map)
        dept_map = get_mappings("department_id", attribute_map)

        N_aisles = len(aisle_map["unique"])
        N_depts = len(dept_map["unique"])
        N_products = bbv_csr.shape[1]

        W_aisle = coo_matrix(
            (np.ones(len(attribute_map)), (attribute_map['col_id'], aisle_map["idx"])), shape=(N_products, N_aisles)
        )
        W_aisle = W_aisle.tocsr()
        bbv_aisle_csr = bbv_csr.dot(W_aisle)

        write_map(aisle_map, "aisle", "hot_map_aisles.parquet", "col_id")
        write_csr_as_coo(bbv_aisle_csr, "hot_baskets_aisles.parquet")

        W_dept = coo_matrix(
            (np.ones(len(attribute_map)), (attribute_map['col_id'], dept_map["idx"])), shape=(N_products, N_depts)
        )
        W_dept = W_dept.tocsr()
        bbv_dept_csr = bbv_csr.dot(W_dept)

        write_map(dept_map, "dept", "hot_map_depts.parquet", "col_id")
        write_csr_as_coo(bbv_dept_csr, "hot_baskets_depts.parquet")

        
        '''CUSTOMER MAPPINGS'''

        print("Aggregating orders into customer vectors via direct COO construction...")

        customer_map = get_mappings("user_id", merged)
        N_rows_cust = len(customer_map["unique"])
        N_cols_cust = bbv_csr.shape[1]

        merged_with_cols = merged.merge(
            product_map_df[['product_id', 'col_id']], 
            on='product_id', 
            how='inner'
        )

        user_id_index = pd.Index(customer_map["unique"])
        customer_row_coords = user_id_index.get_indexer(
        merged_with_cols['user_id']
        )

        # customer_row_coords = merged_with_cols['user_id'].astype('category', categories=customer_map["unique"]).cat.codes
        product_col_coords = merged_with_cols['col_id'].values
        customer_sparse_coo = coo_matrix((np.ones(len(merged_with_cols)), (customer_row_coords, product_col_coords)), shape=(N_rows_cust, N_cols_cust)).tocsr()

        write_map(customer_map, "user", "hot_map_custs.parquet", "row_id")
        write_csr_as_coo(customer_sparse_coo, "hot_customers_products.parquet")

        customer_aisle_csr = customer_sparse_coo.dot(W_aisle)
        write_csr_as_coo(customer_aisle_csr, "hot_customers_aisles.parquet")
        
        customer_dept_csr = customer_sparse_coo.dot(W_dept)
        write_csr_as_coo(customer_dept_csr, "hot_customers_depts.parquet")

        print("process Groceries")

        def read_groceries_csv(basket_option=False):
            from datetime import datetime
            df = pd.read_csv(path / "Groceries_dataset.csv")

            data = {}
            unique = []
            k = 0
            for i in range(len(df)):
                row = dict(df.iloc[i])

                item = row["itemDescription"]
                if item not in unique:
                    unique.append(item)
                
                # key = f"{row["member_number"]}-{date.timetuple().tm_yday}-{date.isocalendar()[0]}" #day/year
                key = f'{row["Member_number"]}'
                if basket_option:
                    date = datetime.strptime(row["Date"], '%d-%m-%Y')
                    if key in data:
                        if date not in data[key]:
                            k += 1
                            data[key][date] = [item]
                        elif item not in data[key][date]:
                            data[key][date].append(item)
                    else:
                        data[key] = {date: [item]}
                else:
                    if key in data:
                        data[key].append(item)
                    else:
                        data[key] = [item]
            
            row_indices = []
            col_indices = []
            data_values = []
            row_map_keys = []
            N_cols = len(unique)

            row_counter = 0
            i = 0
            for key in data:
                if basket_option:
                    for date in data[key]:
                        composite_key = f"{i}"
                        i += 1
                        row_map_keys.append(composite_key)
                        encoded = [0 for _ in range(N_cols)]
                        for item in data[key][date]:
                            encoded[unique.index(item)] += 1
                        for col_idx, count in enumerate(encoded):
                            if count > 0:
                                row_indices.append(row_counter)
                                col_indices.append(col_idx)
                                data_values.append(count)
                        row_counter += 1
                else:
                    composite_key = f"{i}"
                    i += 1 
                    row_map_keys.append(composite_key)
                    encoded = [0 for _ in range(N_cols)]
                    for item in data[key]:
                        encoded[unique.index(item)] += 1
                    for col_idx, count in enumerate(encoded):
                        if count > 0:
                            row_indices.append(row_counter)
                            col_indices.append(col_idx)
                            data_values.append(count)
                    row_counter += 1
                    
            N_rows = row_counter
            row_map_df = pd.Series(row_map_keys, name='order_id').reset_index().rename(columns={'index': 'row_id'})
            sparse_matrix = coo_matrix((data_values, (row_indices, col_indices)), shape=(N_rows, N_cols))
            csr = sparse_matrix.tocsr()
            if basket_option:
                filename = "hot_groceries_baskets.parquet"
                mapname = "hot_map_groceries_baskets.parquet"
                row_map_df = pd.Series(row_map_keys, name='order_id').reset_index().rename(columns={'index': 'row_id'})
            else:
                filename = "hot_groceries_customers.parquet"
                mapname = "hot_map_groceries_customers.parquet"
                row_map_df = pd.Series(row_map_keys, name='user_id').reset_index().rename(columns={'index': 'row_id'})
            
            col_map_df = pd.Series(unique, name=f'product_id').reset_index().rename(columns={'index': 'col_id'})

            if not os.path.isfile(write_path / mapname):
                row_map_df.to_parquet(write_path / mapname)
                print(f"written {mapname}")
            else:
                print(f"skipped {mapname}")

            mapname = "hot_map_groceries_products.parquet"

            if not os.path.isfile(write_path / mapname):
                col_map_df.to_parquet(write_path / mapname)
                print(f"written {mapname}")
            else:
                print(f"skipped {mapname}")
            
            write_csr_as_coo(csr, filename)
        
        read_groceries_csv()
        read_groceries_csv(True)
    else:
        print("parquet files already loaded\n\n")


class CSR_Loader:

    def __init__(self):
        self.path = Path(__file__).parent / "parquet_files"
        self.name_map = {
            "hot_baskets_products.parquet": ["hot_map_products.parquet", "hot_map_orders.parquet"],
            "hot_baskets_aisles.parquet":   ["hot_map_aisles.parquet",   "hot_map_orders.parquet"],
            "hot_baskets_depts.parquet":    ["hot_map_depts.parquet",    "hot_map_orders.parquet"],
            
            "hot_customers_products.parquet": ["hot_map_products.parquet", "hot_map_custs.parquet"],
            "hot_customers_aisles.parquet":   ["hot_map_aisles.parquet",   "hot_map_custs.parquet"],
            "hot_customers_depts.parquet":    ["hot_map_depts.parquet",    "hot_map_custs.parquet"],

            "hot_groceries_baskets.parquet":  
                ["hot_map_groceries_products.parquet", "hot_map_groceries_baskets.parquet"],
            
            "hot_groceries_customers.parquet": 
                ["hot_map_groceries_products.parquet", "hot_map_groceries_customers.parquet"]
        }

        self.orders_map = pd.read_parquet(self.path / "hot_map_orders.parquet")
        self.custs_map = pd.read_parquet(self.path / "hot_map_custs.parquet")
        self.product_names_df = pd.read_csv(Path(__file__).parent / "instacart_data" / "products.csv").set_index("product_id")



    def load(self, filename):
        filename = f"{filename}.parquet"
        df = pd.read_parquet(self.path / filename)
        map_a = pd.read_parquet(self.path / self.name_map[filename][1])
        map_b = pd.read_parquet(self.path / self.name_map[filename][0])
        coo = coo_matrix((df['value'], (df['row_id'], df['col_id'])), shape=(len(map_a), len(map_b)))
        csr = coo.tocsr()
        return csr
    
    def load_reduced_random(self, filename, seed=-1, n=50000):
        csr = self.load(filename)
        if seed > 0:
            np.random.seed(seed)
        
        indices = np.random.choice(csr.shape[0], size=n, replace=False)
        return csr[indices, :], indices
    
    def retrieve_target_information(self, targets, tdata, names=False):

        if "product" in tdata:
            map = "hot_map_products"
        elif "aisle" in tdata:
            map = "hot_map_aisles"
            names = False
        elif "dept" in tdata:
            map = "hot_map_depts"
            names = False
        elif "groceries" in tdata:
            map = "hot_map_groceries_products"
            names = False
        else:
            raise ValueError
        item_map = pd.read_parquet(self.path / f"{map}.parquet")
        item_map_indexed = item_map.set_index("col_id")
        csr = self.load(tdata)
        if "baskets" in tdata:
            # Target is an Order matrix (rows are orders)
            map_df = self.orders_map 
            id_column = 'order_id'
            map_key = 'hot_map_orders.parquet'
        elif "customers" in tdata:
            # Target is a Customer matrix (rows are customers)
            map_df = self.custs_map
            id_column = 'user_id'
            map_key = 'hot_map_custs.parquet'
        else:
            raise ValueError("Target data name must contain 'baskets' or 'customers'.")

        target_rows = map_df[map_df[id_column].isin(targets)]
        row_indices_to_slice = target_rows['row_id'].values
        if len(row_indices_to_slice) == 0:
            print(f"Warning: No matching {id_column}s found in the map.")
            return csr_matrix((0, csr.shape[1])), pd.DataFrame()

        csr_subset = csr[row_indices_to_slice, :]
        # subset_map_df = target_rows.sort_values(by='row_id').reset_index(drop=True)

        items = []
        for i in range(csr_subset.shape[0]):
            sparse_row = csr_subset.getrow(i)
            coo_row = sparse_row.tocoo()

            present_col_ids = coo_row.col          
            item_counts = coo_row.data.astype(int)
            
            if "product" in map:
                tag = "product_id"
            elif "aisle" in map:
                tag = "aisle_id"
            else:
                tag = "dept_id"

            item_ids_present = item_map_indexed.loc[present_col_ids, tag].values
            distribution_series = pd.Series(item_counts, index=item_ids_present)

            if names is False:
                basket_distribution = distribution_series.to_dict()
            else:
                names_series = self.product_names_df.loc[distribution_series.index, 'product_name']
                basket_distribution = pd.Series(distribution_series.values, index=names_series.values).to_dict()
                    
            items.append(basket_distribution)
        return items
    
    def get_cooccurrence_matrix(self, k=120):

        matrix = self.load("hot_baskets_products")
        product_ids = pd.read_parquet(Path(__file__).parent / "parquet_files" / "hot_map_products.parquet")
        product_names = pd.read_csv(Path(__file__).parent / "instacart_data" / "products.csv")

        product_counts = np.array(matrix.sum(axis=0)).flatten()
        top_indices = np.argsort(product_counts)[-k:][::-1]
        top_products = product_ids.iloc[top_indices]["product_id"]
        matrix_filtered = matrix[:, top_indices]
        co_occurrence_top = matrix_filtered.T @ matrix_filtered
        co_occurrence_top.setdiag(0)
        co_occurrence_top.eliminate_zeros()

        id_to_name = product_names.set_index('product_id')['product_name']
        top_names = top_products.map(id_to_name)

        co_occurrence_df = pd.DataFrame(
            co_occurrence_top.toarray(),
            index=top_names.values,
            columns=top_names.values
        )
        return co_occurrence_df, top_names


# for i in range(data.shape[1]):
#     name, x = loader.get_itemset_support_basic([i])
#     item_counts[name[0]] = x
#     print(i, x)
# sorted_counts = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
# print(sorted_counts[:100])
# process_instacart()

# import umap
# from matplotlib import pyplot as plt
# from sklearn.decomposition import TruncatedSVD

# loader = CSR_Loader()
# # data = loader.load_reduced_random("hot_customers_products")

# x = loader.retrieve_target_information([1, 2, 3], "hot_groceries_customers", names=False)

# x = loader.retrieve_target_information([1], "hot_customers_depts", names=False)
# print(x)
# x = loader.retrieve_target_information([1], "hot_customers_aisles", names=False)
# print(x)
# x = loader.retrieve_target_information([i for i in range(1, 10)], "hot_customers_products", names=False)
# for i in range(9):
#     print(x[i])
# input()

# read_instacart_old()


# tsvd = TruncatedSVD(n_components=200)
# print("truncated")
# data = tsvd.fit_transform(data)
# print(data.shape)
# umapper = umap.UMAP(n_jobs=-1)
# embeddings = umapper.fit_transform(data)


# plt.figure(figsize=(10, 8))

# # Create the scatter plot using the first two dimensions
# # Alpha (opacity) is often reduced for large datasets to show density
# plt.scatter(
#     embeddings[:, 0], 
#     embeddings[:, 1], 
#     s=0.5, # Size of points (set small for large datasets)
#     alpha=0.7 # Opacity of points
# )

# plt.gca().set_aspect('equal', 'datalim')
# plt.title("UMAP", fontsize=16)
# plt.xlabel('UMAP Dimension 1')
# plt.ylabel('UMAP Dimension 2')
# plt.show()



class CSR_Reader:

    def __init__(self):
        self.path = Path(__file__).parent / "parquet_files"
        self.prod_map = "hot_map_products.parquet"
        self.order_map = "hot_map_orders.parquet"
        self.filename = "hot_baskets_products.parquet"
        self.product_names_df = pd.read_csv(Path(__file__).parent / "instacart_data" / "products.csv")
        self.user_order_df = pd.read_parquet(self.path / "hot_baskets_products.parquet")
        self.read_sparse_matrix()
    
    def read_sparse_matrix(self):
        sparse_table = pd.read_parquet(self.path / self.filename)
        self.prod_map_df = pd.read_parquet(self.path / self.prod_map)
        self.order_map_df = pd.read_parquet(self.path / self.order_map)

        N_rows = len(self.order_map_df)
        N_cols = len(self.prod_map_df)

        bbv_coo = coo_matrix((sparse_table['value'], (sparse_table['row_id'], sparse_table['col_id'])), shape=(N_rows, N_cols))
        bbv_csr = bbv_coo.tocsr()

        self.csr = bbv_csr

    def retrieve_order_basic(self, order_id, names=False):
        try:
            target_row_id = self.order_map_df[self.order_map_df['order_id'] == order_id]['row_id'].iloc[0]
        except IndexError:
            print(f"Error: Order ID {order_id} not found in the map.")
            return ["None"]
        
        order_vector_sparse = self.csr[target_row_id, :]
        order_vector_coo = order_vector_sparse.tocoo()
        present_col_ids = order_vector_coo.col
        product_list_df = self.prod_map_df[self.prod_map_df['col_id'].isin(present_col_ids)]
        final_product_ids = product_list_df['product_id'].tolist()
        if names:
            products_df = pd.DataFrame(final_product_ids, columns=['product_id'])
            named_products_df = products_df.merge(self.product_names_df[['product_id', 'product_name']], on='product_id', how='left')
            final_product_names = named_products_df['product_name'].tolist()
            return final_product_names
        else:
            return final_product_ids
    
    def retrieve_order_vector(self, order_id):
        try:
            target_row_id = self.order_map_df[self.order_map_df['order_id'] == order_id]['row_id'].iloc[0]
        except IndexError:
            print(f"Error: Order ID {order_id} not found in the map.")
            return None
        return self.csr[target_row_id, :]

    def retrieve_customer_basic(self, customer_id, names=False, type="list"):
        customer_orders = self.user_order_df[(self.user_order_df['user_id'] == customer_id)]
        order_ids_to_sum = customer_orders['order_id'].tolist()
        customer_vector_sum = []
        for order_id in order_ids_to_sum:
            order_list = self.retrieve_order_basic(order_id=order_id, names=names)
            if type=="list":
                customer_vector_sum += order_list
            if type=="array":
                customer_vector_sum.append({order_id: order_list})
        return customer_vector_sum
    

    def retrieve_customer_vector(self, customer_id):
        customer_orders = self.user_order_df[(self.user_order_df['user_id'] == customer_id)]
        order_ids_to_sum = customer_orders['order_id'].tolist()
        N_cols = self.csr.shape[1]
        customer_vector_sum = csr_matrix((1, N_cols), dtype=self.csr.dtype)
        for order_id in order_ids_to_sum:
            try:
                order_vector = self.retrieve_order_vector(order_id)
                customer_vector_sum = customer_vector_sum + order_vector
            except:
                pass
        return customer_vector_sum