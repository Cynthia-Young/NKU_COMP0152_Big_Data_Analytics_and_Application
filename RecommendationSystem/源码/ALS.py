import numpy as np
from datapreprocess import *
import gc

def calculate_rmse(rating_matrix, prediction_matrix):
    mask = rating_matrix > 0 
    mse = np.sum((rating_matrix[mask] - prediction_matrix[mask]) ** 2) / np.sum(mask)
    rmse = np.sqrt(mse)
    return rmse


def als(train_data, num_factors=10, num_iterations=10, regularization=0.1):
    users = list(train_data.keys())
    items = {item for user_items in train_data.values() for item, _ in user_items}
    num_users = len(users)
    num_items = len(items)

    user_to_index = {user: idx for idx, user in enumerate(users)}
    item_to_index = {item: idx for idx, item in enumerate(items)}

    rating_matrix = np.zeros((num_users, num_items))
    for user, user_items in train_data.items():
        for item, rating in user_items:
            rating_matrix[user_to_index[user], item_to_index[item]] = rating

    user_matrix = np.random.normal(size=(num_users, num_factors))
    item_matrix = np.random.normal(size=(num_items, num_factors))
    
    for iteration in range(num_iterations):
        for u in range(num_users):
            user_ratings = rating_matrix[u, :]
            non_zero_items = user_ratings.nonzero()[0]
            item_submatrix = item_matrix[non_zero_items, :]
            ratings_submatrix = user_ratings[non_zero_items]
            A = item_submatrix.T @ item_submatrix + regularization * np.eye(num_factors)
            b = item_submatrix.T @ ratings_submatrix
            user_matrix[u, :] = np.linalg.solve(A, b)

        for i in range(num_items):
            item_ratings = rating_matrix[:, i]
            non_zero_users = item_ratings.nonzero()[0]
            user_submatrix = user_matrix[non_zero_users, :]
            ratings_submatrix = item_ratings[non_zero_users]
            A = user_submatrix.T @ user_submatrix + regularization * np.eye(num_factors)
            b = user_submatrix.T @ ratings_submatrix
            item_matrix[i, :] = np.linalg.solve(A, b)

        prediction_matrix = user_matrix @ item_matrix.T
        rmse = calculate_rmse(rating_matrix, prediction_matrix)
        print(f"Iteration {iteration + 1}/{num_iterations}, RMSE: {rmse}")

        # 清理不再需要的内存并调用垃圾回收
        del prediction_matrix
        gc.collect()

    return user_matrix, item_matrix, user_to_index, item_to_index

def predict(user_id, item_id, user_matrix, item_matrix, user_to_index, item_to_index):
    user_idx = user_to_index.get(user_id)
    item_idx = item_to_index.get(item_id)
    if user_idx is not None and item_idx is not None:
        return user_matrix[user_idx, :].dot(item_matrix[item_idx, :])
    else:
        return None


if __name__ == '__main__':
    train_dict = read_train_data('data/train.txt')
    user_matrix, item_matrix, user_to_index, item_to_index = als(train_dict)

    test_dict = read_test_data('data/test.txt')
    predictions = {}
    for user_id, item_ids in test_dict.items():
        predictions[user_id] = [[item_id, predict(user_id, item_id, user_matrix, item_matrix, user_to_index, item_to_index)] for item_id in item_ids]
        
        
    output_file = 'result/ALS_predictions.txt'
    with open(output_file, 'w') as f:
        for user_id, items in predictions.items():
            f.write(f"{user_id}|{len(items)}\n")
            for item_id, rating in items:
                f.write(f"{item_id} {rating}\n")

    print(f"Predictions saved to {output_file}")
    
    input("please input any key to exit!")