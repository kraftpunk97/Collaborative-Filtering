import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

max_neighbors = 40

train_df = pd.read_csv('netflix/TrainingRatings.txt', names=['MovieID', 'UserID', 'Rating'])
train_df = train_df.astype({'Rating': 'int64'})
test_df = pd.read_csv('netflix/TestingRatings.txt', names=['MovieID', 'UserID', 'Rating'])
test_df = test_df.astype({'Rating': 'int64'})

train_pivot = train_df.pivot(index='UserID', columns='MovieID', values='Rating')
train_pivot_0 = train_pivot.fillna(0)

running_mae = 0
running_rmse = 0
running_n = 0

# Processing test example each unique test movie at a time, to eliminate redundant model training.
for movie in test_df['MovieID'].unique():
    active_users = test_df[test_df['MovieID'] == movie]['UserID']

    # Only considering users that have rated the movie in consideration.
    ratings4movie = train_pivot_0[train_pivot_0[movie] != 0]
    model_kNN = NearestNeighbors(metric='cosine', algorithm='brute')
    model_kNN.fit(ratings4movie)

    n_neighbors = min(len(ratings4movie), max_neighbors)
    _, indices_arr = model_kNN.kneighbors(train_pivot_0.loc[test_df[test_df['MovieID'] == movie]['UserID']].to_numpy(),
                                          n_neighbors=n_neighbors)
    neighbors = ratings4movie.index[indices_arr]

    neighbor_ratings = np.stack([train_pivot_0.loc[neighbor_list].to_numpy()
                                 for neighbor_list in neighbors])
    active_user_rating = train_pivot_0.loc[active_users]
    active_user_rating = np.repeat(active_user_rating.to_numpy()[:, np.newaxis, :], n_neighbors, axis=1)

    # Only considering movies that have been rated by both, active_user and its neighbor.
    active_neighbor_common = (active_user_rating * neighbor_ratings).astype(bool).astype(int)

    active_user_rating_means = active_user_rating.mean(axis=2)
    neighbor_rating_means = neighbor_ratings.mean(axis=2)

    neighbor_rating_diff = (neighbor_ratings - neighbor_rating_means[:, :, np.newaxis]) * active_neighbor_common
    active_user_rating_diff = (active_user_rating - active_user_rating_means[:, :, np.newaxis]) * active_neighbor_common

    numerator = np.sum(neighbor_rating_diff * active_user_rating_diff, axis=2)
    denominator = np.sqrt(np.sum(active_user_rating_diff**2, axis=2) * np.sum(neighbor_rating_diff ** 2, axis=2))
    pearson_weights = numerator / (denominator + 1)  # Adding 1 to denominator to avoid division by zero.
    pearson_weights /= pearson_weights.sum(axis=1)[:, np.newaxis]

    net_diff = pearson_weights * (np.stack(
        [train_pivot_0[movie].loc[passive_user_list].to_numpy() for passive_user_list in
         neighbors]) - neighbor_rating_means)
    pred_rating = net_diff.sum(axis=1) + train_pivot_0.loc[active_users].mean(axis=1)

    true_rating = test_df[test_df['MovieID'] == movie]['Rating'].to_numpy()
    pred_rating = pred_rating.to_numpy()
    rmse = np.sum((pred_rating - true_rating)**2)
    mae = np.sum(np.abs(pred_rating - true_rating))

    running_rmse = np.sqrt(((running_rmse**2)*running_n + rmse) / (running_n+len(true_rating)))
    running_mae = (running_mae*running_n + mae) / (running_n+len(true_rating))
    running_n += len(true_rating)

    print("Processed {}% of the testing dataset...".format(running_n*100/len(test_df)))
    print("Current RMSE = {}".format(running_rmse))
    print("Current MAE = {}".format(running_mae))
    print()
    print()

print("Final results:")
print("RMSE = {}".format(running_rmse))
print("MAE = {}".format(running_mae))


