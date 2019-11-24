"""Process user-item pairs into sparce ratings matrix"""

import os
import pickle

import numpy as np
from scipy.sparse import save_npz, lil_matrix
from tqdm import tqdm
from typing import Dict, List, Tuple


def process_data(input_path: str, sep: str = ','):
    """
    Read the data from input path and process into ratings matrix.

    Args:
        input_path (str): Path to input file.
        sep (str): Separator inside the file.
    Returns:
        ratings (scipy.sparce.csr_matrix): 2d sparse matrix with user-item ratings.
        user_to_idx (dict): user to index in ratings matrix correspondance
        item_to_idx (dict): item to index in ratings matrix correspondance

    """
    user_to_idx: Dict[str, int] = {}
    item_to_idx: Dict[str, int] = {}
    data: List[Tuple[int, int, float]] = []

    user_num, item_num = 0, 0

    with open(input_path, 'r') as file:
        # assign indexes to users and items
        for line in tqdm(file, desc='[INFO] Reading ratings data'):
            if line:
                user, item, rating, _ = line.rstrip().split(sep=sep)
                if user not in user_to_idx:
                    user_to_idx[user] = user_num
                    user_num += 1
                if item not in item_to_idx:
                    item_to_idx[item] = item_num
                    item_num += 1

                data.append((user_to_idx[user], item_to_idx[item], float(rating)))

    ratings = lil_matrix((user_num, item_num), dtype=np.float32)
    for user_idx, item_idx, rating in tqdm(data, desc='[INFO] Assigning values to sparce matrix'):
        ratings[user_idx, item_idx] = rating

    return ratings.tocsr(), user_to_idx, item_to_idx


if __name__ == '__main__':
    filename = 'ratings_Books.csv'
    input_dir = os.path.join('..', '..', 'data', 'subsample')
    output_dir = os.path.join('..', '..', 'data', 'processed', 'sparce')

    ratings, user_to_idx, item_to_idx = process_data(os.path.join(input_dir, filename))

    save_npz(os.path.join(output_dir, 'ratings.npz'), ratings)

    with open(os.path.join(output_dir, 'user_to_idx.pickle'), 'wb') as file:
        pickle.dump(user_to_idx, file)

    with open(os.path.join(output_dir, 'item_to_idx.pickle'), 'wb') as file:
        pickle.dump(item_to_idx, file)

    print(f'[INFO] Number of ratings: {ratings.nnz}',
          f'[INFO] Shape of matrix: {ratings.shape}',
          f'[INFO] Sparcity: {ratings.nnz / ratings.shape[0] / ratings.shape[1] * 100}%', sep='\n')
