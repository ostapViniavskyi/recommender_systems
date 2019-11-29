import numpy as np

from .user_based import UserBased


class ItemBased:
    def __init__(self, M, N, neighbors=20, min_common_users=5, min_values=1.0, max_value=5.0):
        self.ubcf = UserBased(N, M, neighbors, min_common_users, min_values, max_value)

    def fit(self, train_ratings, item_to_users):
        train_ratings_reversed = {
            (i, u): r for (u, i), r in train_ratings.items()
        }
        self.ubcf.fit(train_ratings_reversed, item_to_users)
        return self

    def predict(self, i, k):
        return self.ubcf.predict(k, i)

    def score(self, test_ratings):
        """Return RMSE for given test set"""
        rmse = 0
        for (i, k), y_true in test_ratings.items():
            y_pred = self.predict(i, k)
            rmse += (y_pred - y_true) ** 2
        return np.sqrt(rmse / len(test_ratings))
