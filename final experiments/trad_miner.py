from PAMI.frequentPattern.basic import FPGrowth as fpgrowth
from instacart_loader import CSR_Loader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

l = CSR_Loader()

NSGA_data = pd.read_csv("NSGA_rules.csv")

# data = l.load("hot_baskets_products")
data, indices = l.load_reduced_random("hot_baskets_products", seed=42, n=10000)
n = data.sum()

result = []
# for i in range(200, 10000, 100):
#     minsup = round(n * (1 / i))
#     col_sums = data.sum(axis=0).A.flatten()
#     cols_to_keep = col_sums > minsup


#     divdata = data[:, cols_to_keep]
#     result.append(((1 / i), divdata.shape[1]))

# minsup_values = [x[0] for x in result]
# num_items = [x[1] for x in result]

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(minsup_values, num_items, marker='o')
# plt.xlabel('Minimum Support (fraction)')
# plt.ylabel('Number of Items')
# plt.title('Number of Items vs Minimum Support')
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

minsup = round(n * .00005)
col_sums = data.sum(axis=0).A.flatten()
cols_to_keep = col_sums > minsup
original_indices = np.where(cols_to_keep)[0]

data_filtered = data[:, cols_to_keep].A
transactions = []
for i in range(data_filtered.shape[0]):
    # Get column indices where value is 1, map to original indices
    items = [str(original_indices[j]) for j in range(data_filtered.shape[1]) if data_filtered[i, j] == 1]
    transactions.append(' '.join(items))  # Join with space

# Create DataFrame with "Transactions" column
data_df = pd.DataFrame({'Transactions': transactions})

print(f"DataFrame shape: {data_df.shape}")
print(f"Example transaction: {data_df['Transactions'].iloc[0]}")
print(f"Number of kept columns: {len(original_indices)}")

# Run FPGrowth to find frequent itemsets
obj = fpgrowth.FPGrowth(iFile=data_df, minSup=minsup, sep=' ')
obj.mine()
patterns = obj.getPatterns()

print(f"Found {len(patterns)} frequent itemsets")

# Generate association rules from patterns
# We need to manually calculate confidence and lift
from itertools import combinations

rules = []
n_transactions = data_filtered.shape[0]

for itemset, support in patterns.items():
    # itemset is already a tuple, not a string
    items = list(itemset) if isinstance(itemset, tuple) else [itemset]
    
    if len(items) > 1:
        # Generate rules for all possible splits
        for i in range(1, len(items)):
            for antecedent_items in combinations(items, i):
                antecedent = tuple(sorted(antecedent_items))
                consequent_items = tuple(item for item in items if item not in antecedent_items)
                consequent = tuple(sorted(consequent_items))
                
                # Calculate metrics
                if antecedent in patterns:
                    support_rule = support
                    confidence = support / patterns[antecedent]
                    
                    # Calculate lift
                    if consequent in patterns:
                        support_consequent = patterns[consequent] / n_transactions
                        lift = confidence / support_consequent
                        
                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support_rule,
                            'confidence': confidence,
                            'lift': lift,
                            'length': len(items)  # Total itemset length
                        })

print(f"Generated {len(rules)} association rules")

# Extract metrics from FPGrowth rules
supports = [rule['support'] for rule in rules]
confidences = [rule['confidence'] for rule in rules]
lifts = [rule['lift'] for rule in rules]
lengths = [rule['length'] for rule in rules]

# Extract metrics from NSGA data
nsga_supports = NSGA_data['support'].values
nsga_confidences = NSGA_data['confidence'].values
nsga_lifts = NSGA_data['lift'].values
nsga_lengths = []
for idx, row in NSGA_data.iterrows():
    antecedent_items = row['antecedent'].split() if isinstance(row['antecedent'], str) else []
    consequent_items = row['consequent'].split() if isinstance(row['consequent'], str) else []
    total_length = len(antecedent_items) + len(consequent_items)
    nsga_lengths.append(total_length)

# Create 3D plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot FPGrowth rules (support, confidence, lift)
scatter1 = ax.scatter(np.log(supports), confidences, lengths, 
                      c='blue', s=50, alpha=0.6, label='FPGrowth Rules')

# Plot NSGA data (support, confidence, lift)
scatter2 = ax.scatter(np.log(nsga_supports), nsga_confidences, nsga_lengths,
                      c='red', s=50, alpha=0.6, label='NSGA Data')

ax.set_xlabel('Support')
ax.set_ylabel('Confidence')
ax.set_zlabel('Length')
ax.set_title('Association Rules: FPGrowth vs NSGA')
ax.legend()
plt.tight_layout()
plt.show()
