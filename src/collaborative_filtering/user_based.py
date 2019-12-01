import numpy as np
from tqdm import tqdm


class UserBased:
    def __init__(self, M, N, neighbors=20, min_common_items=5, min_values=1.0, max_value=5.0):
        self.M = M
        self.N = N
        self.neighbors = neighbors
        self.min_common_items = min_common_items
        self.min_value = min_values
        self.max_value = max_value

        self.neighbors_weights = np.empty((M,), dtype=list)
        self.user_deviations = np.empty((M,), dtype=dict)

    def fit(self, train_ratings, user_to_items):
        """Fit user-base collaborative filtering model"""
        # set user_mean with default value of global mean
        self.user_means = np.full(
            (self.M,),
            fill_value=sum(train_ratings.values()) / len(train_ratings.values())
        )

        neighbours_correlation = np.empty((self.M,))
        for i in range(self.M):

            if i not in user_to_items:
                self.neighbors_weights[i] = []
                continue

            items_i = set(user_to_items[i])
            ratings_i = {
                item: train_ratings[(i, item)] for item in items_i
            }
            mean_i = np.mean(list(ratings_i.values()))
            deviation_i = {
                item: (rating - mean_i) for item, rating in ratings_i.items()
            }
            self.user_means[i] = mean_i
            self.user_deviations[i] = deviation_i

            for j in range(self.M):
                if i == j or j not in user_to_items:  # can't include itself as neighbor
                    neighbours_correlation[j] = 0
                    continue

                items_j = set(user_to_items[j])
                common = list(items_i.intersection(items_j))

                if len(common) < self.min_common_items:  # don't include users that have to few items in common
                    neighbours_correlation[j] = 0
                    continue

                ratings_j = {
                    item: train_ratings[(j, item)] for item in items_j
                }
                mean_j = np.mean(list(ratings_j.values()))
                deviation_j = {
                    item: (rating - mean_j) for item, rating in ratings_j.items()
                }

                # correlation between user i and j
                common_dev_i = np.array([deviation_i[k] for k in common])
                common_dev_j = np.array([deviation_j[k] for k in common])
                neighbours_correlation[j] = \
                    np.dot(common_dev_i, common_dev_j) / np.linalg.norm(common_dev_i) / np.linalg.norm(common_dev_j)

            top_k_idx = np.argpartition(-np.abs(neighbours_correlation), self.neighbors)[:self.neighbors]
            top_k_idx = [k for k in top_k_idx if neighbours_correlation[k] != 0]
            self.neighbors_weights[i] = [
                (j, neighbours_correlation[j]) for j in top_k_idx if neighbours_correlation[j] != -np.inf
            ]
        return self

    def predict(self, i, k):
        """Predict score(i, k)"""
        neighbours = self.neighbors_weights[i]
        weighted_deviations, weights = 0, 0
        for j, c_ij in neighbours:
            if k in self.user_deviations[j]:
                weighted_deviations += c_ij * self.user_deviations[j][k]
                weights += np.abs(c_ij)

        # if no neighbors are found, predict the mean for that user
        # also if neighbors didn't bought item k - predict mean
        if weights != 0:
            score = self.user_means[i] + weighted_deviations / weights
        else:
            score = self.user_means[i]
        if score < self.min_value:
            return self.min_value
        return min(score, self.max_value)

    def score(self, test_ratings):
        """Return RMSE for given test set"""
        rmse = 0
        for (i, k), y_true in test_ratings.items():
            y_pred = self.predict(i, k)
            rmse += (y_pred - y_true) ** 2
        return np.sqrt(rmse / len(test_ratings))
