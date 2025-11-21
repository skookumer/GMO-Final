import pandas as pd
import numpy as np

from instacart_loader import CSR_Loader
from instacart_loader import process_parquet

from sklearn.decomposition import TruncatedSVD


'''download the instacart dataset and Groceries and put it all in instacart_data'''

process_parquet()
loader = CSR_Loader()

'''
instacart dataset file codes:

    tdata="hot_X_Y"

    where X = customers / baskets
    and Y =   products / aisles / depts

    names = optional, returns product names instead of ids

groceries dataset file codes:

    tdata="hot_groceries_X"

    where X = customers / baskets

Groceries doesn't have a Y option and always returns names
'''


'''EXAMPLE FUNCTIONS'''

#retrieve item counts for customers 1, 2, 3 (for cluster histograms etc)
x = loader.retrieve_target_information([1, 2, 3], "hot_groceries_customers", names=False)
print(x, "\n\n")

#retrieve the department location for just customer 1's items
x = loader.retrieve_target_information([1], "hot_customers_depts", names=False)
print(x, "\n\n")

#retrieve aisle locations for products in baskets 100-102
x = loader.retrieve_target_information([100,101,102], "hot_baskets_aisles", names=False)
print(x, "\n\n")

#get product names for items purchased by customers 1-10
x = loader.retrieve_target_information([i for i in range(1, 10)], "hot_customers_products", names=True)
for i in range(9):
    print(x[i])

'''
The following functions return sparse csr matrices which work with some functions and not others.
They should work with most clustering but can be converted to dense numpy arrays to work with anything.
'''

#load groceries
data = loader.load("hot_groceries_baskets")
print("shape:", data.shape, "size:", data.size)

#get the full 200,000 x 50,000 sparse customer-products CSR matrix
data = loader.load("hot_customers_products")
print("shape:", data.shape, "size:", data.size)

#get the 3,346,083 x 134 sparse baskets-aisles CSR matrix
data = loader.load("hot_baskets_aisles")
print("shape:", data.shape, "size:", data.size)

#get 49,999 random samples from the customer_products matrix
data = loader.load_reduced_random("hot_customers_products", seed=42, n=49999)
print("shape:", data.shape, "size:", data.size)

#Truncated SVD is computationally efficient because it does not require centering the data (which destroys the sparsity)
#UMAP might take 3-hr even on the reduced customers-aisles matrix

tsvd = TruncatedSVD(n_components=200)
data_reduced = tsvd.fit_transform(data)

#use this for downstream tasks...