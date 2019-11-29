"""Process user-item pairs into sparce ratings matrix"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, lil_matrix


def process_data(input_path: str, sep: str = ',', test_size=0.2):
    """
    Read the data from input path and process into ratings matrix.

    Args:
        input_path (str): Path to input file.
        sep (str): Separator inside the file.
        test_size (float): Part of dataset to be used for testing.
    Returns:
        train_ratings (scipy.sparse.csr_matrix): 2d sparse matrix with user-item ratings.
        test_ratings (scipy.sparse.csr_matrix): 2d sparse matrix with user-item ratings.
        user_to_idx (dict): user to index in ratings matrix correspondence
        item_to_idx (dict): item to index in ratings matrix correspondence

    """

    df = pd.read_csv(input_path, names=['user', 'item', 'rating', 'time'], sep=sep)

    df.drop(columns=['time'], inplace=True)

    unique_users = df['user'].unique()
    M = unique_users.shape[0]
    print(f'[INFO] Number of unique users: {len(unique_users)}')
    user_to_idx = dict(zip(unique_users, np.arange(M)))

    unique_items = df['item'].unique()
    N = unique_items.shape[0]
    print(f'[INFO] Number of unique items: {len(unique_items)}')
    item_to_idx = dict(zip(unique_items, np.arange(N)))

    df['user'] = df['user'].map(user_to_idx)
    df['item'] = df['item'].map(item_to_idx)

    # randomly shuffle data and split for train/test
    df_idx = np.arange(df.shape[0])
    np.random.shuffle(df_idx)
    df_train = df.iloc[df_idx[:-int(test_size * df.shape[0])]]
    df_test = df.iloc[df_idx[-int(test_size * df.shape[0]):]]

    train_ratings = lil_matrix((M, N))
    test_ratings = lil_matrix((M, N))

    for row in df_train.itertuples():
        _, user, item, rating = row
        train_ratings[user, item] = rating

    for row in df_test.itertuples():
        _, user, item, rating = row
        test_ratings[user, item] = rating

    return train_ratings.tocsr(), test_ratings.tocsr(), user_to_idx, item_to_idx


if __name__ == '__main__':
    filename = 'ratings_Books.csv'
    input_dir = os.path.join('..', '..', 'data', 'subsample')
    output_dir = os.path.join('..', '..', 'data', 'processed', 'sparse')

    train_ratings, test_ratings, user_to_idx, item_to_idx = process_data(
        os.path.join(input_dir, filename),
        sep=',', test_size=0.2
    )

    save_npz(os.path.join(output_dir, 'train_ratings.npz'), train_ratings)
    save_npz(os.path.join(output_dir, 'test_ratings.npz'), test_ratings)

    with open(os.path.join(output_dir, 'user_to_idx.pickle'), 'wb') as file:
        pickle.dump(user_to_idx, file)

    with open(os.path.join(output_dir, 'item_to_idx.pickle'), 'wb') as file:
        pickle.dump(item_to_idx, file)

    print(f'[INFO] Number of ratings in train set: {train_ratings.nnz}',
          f'[INFO] Sparcity of train matrix: '
          f'{train_ratings.nnz / train_ratings.shape[0] / train_ratings.shape[1] * 100}%', sep='\n')
