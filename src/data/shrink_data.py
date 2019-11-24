"""Select subsample of raw data"""

import os
import pandas as pd
from collections import Counter

filename = 'ratings_Books.csv'
input_dir = os.path.join('..', '..', 'data', 'raw')
output_dir = os.path.join('..', '..', 'data', 'subsample')

data = pd.read_csv(
    os.path.join(input_dir, filename),
    header=None,
    names=['user', 'item', 'score', 'timestamp']
)

M = data['user'].nunique()  # number of unique users
N = data['item'].nunique()  # number of unique items
print(f'[INFO] Unique users: {M}, unique items: {N}')

m, n = 10000, 2000  # number of users and items to keep
users = [i for i, _ in Counter(data['user']).most_common(m)]
items = [i for i, _ in Counter(data['item']).most_common(n)]

subset = data[data['user'].isin(users) & data['item'].isin(items)].copy()
print(f'[INFO] Subset shape: {subset.shape}')
subset.to_csv(os.path.join(output_dir, filename), index=False, header=False)
