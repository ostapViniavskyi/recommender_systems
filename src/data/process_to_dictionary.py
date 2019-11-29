"""Process user-item pairs into dictionary"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def process_data(input_path: str, sep: str = ',', test_size=0.2):
    """
    Read the data from input path and process into ratings dictionary.

    Args:
        input_path (str): Path to input file.
        sep (str): Separator inside the file.
        test_size (float): Part of dataset to be used for testing.
    Returns:
        train_ratings (dict(tuple(int, int), float)): dictionary of train user-item ratings.
        test_ratings (dict(tuple(int, int), float)): dictionary of test user-item ratings.
        user_to_idx (dict): user to index in ratings matrix correspondence.
        item_to_idx (dict): item to index in ratings matrix correspondence.
        user_to_items (dict): dictionary that represents which items user reviewed.
        item_to_users (dict): dictionary that represents which users reviewed item.
    """
    user_to_items: Dict[int, List[int]] = {}
    item_to_users: Dict[int, List[int]] = {}
    train_ratings: Dict[Tuple[int, int], float] = {}
    test_ratings: Dict[Tuple[int, int], float] = {}

    df = pd.read_csv(input_path, names=['user', 'item', 'rating', 'time'], sep=sep)

    df.drop(columns=['time'], inplace=True)

    unique_users = df['user'].unique()
    print(f'[INFO] Number of unique users: {len(unique_users)}')
    user_to_idx = dict(zip(unique_users, np.arange(unique_users.shape[0])))

    unique_items = df['item'].unique()
    print(f'[INFO] Number of unique items: {len(unique_items)}')
    item_to_idx = dict(zip(unique_items, np.arange(unique_items.shape[0])))

    df['user'] = df['user'].map(user_to_idx)
    df['item'] = df['item'].map(item_to_idx)

    # randomly shuffle data and split for train/test
    df_idx = np.arange(df.shape[0])
    np.random.shuffle(df_idx)
    df_train = df.iloc[df_idx[:-int(test_size * df.shape[0])]]
    df_test = df.iloc[df_idx[-int(test_size * df.shape[0]):]]
    # process training set
    for row in df_train.itertuples():
        _, user, item, rating = row

        if user not in user_to_items:
            user_to_items[user] = []
        if item not in item_to_users:
            item_to_users[item] = []

        user_to_items[user].append(item)
        item_to_users[item].append(user)

        train_ratings[(user, item)] = rating
    # save test set in dictionary as well
    for row in df_test.itertuples():
        _, user, item, rating = row
        test_ratings[(user, item)] = rating

    return train_ratings, test_ratings, user_to_items, item_to_users, user_to_idx, item_to_idx


if __name__ == '__main__':
    filename = 'ratings_Books.csv'
    input_dir = os.path.join('..', '..', 'data', 'subsample')
    output_dir = os.path.join('..', '..', 'data', 'processed', 'filtering')

    train_ratings, test_ratings, user_to_items, item_to_users, user_to_idx, item_to_idx = process_data(
        os.path.join(input_dir, filename),
        sep=',', test_size=0.2
    )

    with open(os.path.join(output_dir, 'user_to_idx.pickle'), 'wb') as file:
        pickle.dump(user_to_idx, file)

    with open(os.path.join(output_dir, 'item_to_idx.pickle'), 'wb') as file:
        pickle.dump(item_to_idx, file)

    with open(os.path.join(output_dir, 'user_to_items.pickle'), 'wb') as file:
        pickle.dump(user_to_items, file)

    with open(os.path.join(output_dir, 'item_to_users.pickle'), 'wb') as file:
        pickle.dump(item_to_users, file)

    with open(os.path.join(output_dir, 'train_ratings.pickle'), 'wb') as file:
        pickle.dump(train_ratings, file)

    with open(os.path.join(output_dir, 'test_ratings.pickle'), 'wb') as file:
        pickle.dump(test_ratings, file)

    print(f'[INFO] Number of ratings in train set: {len(train_ratings)}',
          f'[INFO] Number of users in train set: {len(user_to_items)}',
          f'[INFO] Number of items in train set: {len(item_to_users)}',
          f'[INFO] Sparcity of train set: {(len(train_ratings) / len(user_to_items) / len(item_to_users) * 100):.3f} %',
          sep='\n')
