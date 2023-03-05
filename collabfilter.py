import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

train_df = pd.read_csv('netflix/TrainingRatings.txt', names=['MovieID', 'UserID', 'Rating'])
train_df = train_df.astype({'Rating': 'int64'})
test_df = pd.read_csv('netflix/TestingRatings.txt', names=['MovieID', 'UserID', 'Rating'])
test_df = test_df.astype({'Rating': 'int64'})

train_pivot = train_df.pivot(index='UserID', columns='MovieID', values='Rating')
train_pivot_0 = train_pivot.fillna(0)

running_mae = 0
running_rmse = 0
running_n = 0

for movie in test_df['MovieID'].unique():
    ratings4movie = train_pivot_0[train_pivot_0[movie] != 0]
    model_kNN = NearestNeighbors(metric='cosine', algorithm='brute')
    model_kNN.fit(ratings4movie)

    true_rating = test_df[test_df['MovieID'] == movie]['Rating']

    n_neighbors = min(len(ratings4movie), 40)
    _, indices_arr = model_kNN.kneighbors(train_pivot_0.loc[test_df[test_df['MovieID'] == movie]['UserID']].to_numpy(),
                                          n_neighbors=n_neighbors)
    passive_users = ratings4movie.index[indices_arr]
    active_users = test_df[test_df['MovieID'] == movie]['UserID']

    passive_user_rating = np.stack([train_pivot_0.loc[passive_user_list].to_numpy() for passive_user_list in passive_users])
    active_user_rating = train_pivot_0.loc[active_users]
    active_user_rating = np.repeat(active_user_rating.to_numpy()[:, np.newaxis, :], n_neighbors, axis=1)

    active_passive_common = (active_user_rating * passive_user_rating).astype(bool).astype(int)

    active_user_rating_means = active_user_rating.mean(axis=2)
    passive_user_rating_means = passive_user_rating.mean(axis=2)

    passive_user_rating_diff = (passive_user_rating - passive_user_rating_means[:, :, np.newaxis]) * active_passive_common
    active_user_rating_diff = (active_user_rating - active_user_rating_means[:, :, np.newaxis]) * active_passive_common

    numerator = np.sum(passive_user_rating_diff * active_user_rating_diff, axis=2)
    denominator = np.sqrt(np.sum(active_user_rating_diff**2, axis=2) * np.sum(passive_user_rating_diff**2, axis=2)) + 1
    pearson_weights = numerator / denominator
    pearson_weights /= pearson_weights.sum(axis=1)[:, np.newaxis]

    net_diff = pearson_weights * (np.stack(
        [train_pivot_0[movie].loc[passive_user_list].to_numpy() for passive_user_list in
         passive_users]) - passive_user_rating_means)
    pred_rating = net_diff.sum(axis=1) + train_pivot_0.loc[active_users].mean(axis=1)

    rmse = (pred_rating - true_rating)**2
    mae = (pred_rating - true_rating)

    running_rmse = ((running_rmse**2)*running_n + rmse) / (running_n+len(true_rating))
    running_mae = (running_mae*running_n + mae) / (running_n+len(true_rating))
    running_n += len(true_rating)

    print("Processed {}% of the testing dataset...".format(running_n/len(test_df)))
    print("Current RMSE = {}".format(running_rmse))
    print("Current MAE = {}".format(running_mae))
    print()
    print()

print("Final results:")
print("RMSE = {}".format(running_rmse))
print("MAE = {}".format(running_mae))


