import pandas as pd
import numpy as np
import random
import csv
import matplotlib.pyplot as plt

def calculate_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound

def remove_outliers_iqr(data):
    lower_bound, upper_bound = calculate_iqr(data)
    return data[(data >= lower_bound) & (data <= upper_bound)]

def integer_encode(X, columns):
    for column in columns:
        X[column] = X[column].astype('category').cat.codes
    return X

def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    test_samples = int(n_samples * test_size)
    test_indices = np.random.choice(n_samples, test_samples, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=19, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._mean_value(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, n_feats, replace=False)

        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return Node(value=self._mean_value(y))

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thresh in thresholds:
                gain = self._information_gain(y, X_column, thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_variance = np.var(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = np.var(y[left_idxs]), np.var(y[right_idxs])
        child_variance = (n_l / n) * e_l + (n_r / n) * e_r

        ig = parent_variance - child_variance
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.where(X_column <= split_thresh)[0]
        right_idxs = np.where(X_column > split_thresh)[0]
        return left_idxs, right_idxs

    def _mean_value(self, y):
        return np.mean(y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            if node.value is None:
                return np.mean(self.y_train)
            else:
                return node.value

        if node.threshold is not None:
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            return node.value
    
    
def rubles_to_usd(amount_in_rubles, exchange_rate=0.01090635):
    amount_in_usd = amount_in_rubles * exchange_rate
    return amount_in_usd

results = {'i': [], 'MSE': [], 'MAE': [], 'RMSE': [], 'Avg_percent_error': []}

for i in range(1, 51):
    #regular testing
    # # Load and preprocess data
    # data = pd.read_csv('data.csv')
    # # Assuming data preprocessing like one-hot encoding has already been done
    # data = integer_encode(data, ['Apartment_type', 'Metro_station', 'Region', 'Renovation'])
    # # Split data into features and target variable
    # X = data.drop(columns=['Price']).values
    # y = data['Price'].values

    # X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

    # decision_tree = DecisionTree(max_depth=i, min_samples_split=7)
    # decision_tree.fit(X_train, y_train)

    # # Make predictions on the test set
    # predictions = decision_tree.predict(X_test)

    # # Calculate Mean Squared Error (MSE)
    # mse = np.mean(rubles_to_usd(predictions - y_test) ** 2)
    # mse = rubles_to_usd(mse)

    # # Calculate Mean Absolute Error (MAE)
    # mae = np.mean(np.abs(predictions - y_test))
    # mae = rubles_to_usd(mae)

    # # Calculate Root Mean Squared Error (RMSE)
    # rmse = np.sqrt(mse)

    # # Calculate average percentage error
    # avg_percent_error = np.mean(np.abs((predictions - y_test) / y_test)) * 100

    # # Append results to lists
    # results['i'].append(i)
    # results['MSE'].append(mse)
    # results['MAE'].append(mae)
    # results['RMSE'].append(rmse)
    # results['Avg_percent_error'].append(avg_percent_error)

    # print("i:", i)
    # print("Mean Squared Error (MSE):", mse)
    # print("Mean Absolute Error (MAE):", mae)
    # print("Root Mean Squared Error (RMSE):", rmse)
    # print("Average Percentage Error:", avg_percent_error, "%")

    #iqr testing
    print(i)
    data = pd.read_csv('data.csv')
    data = integer_encode(data, ['Apartment_type', 'Metro_station', 'Region', 'Renovation'])

    X = data.drop(columns=['Price']).values
    y = data['Price'].values

    X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)

    random_forest = DecisionTree(max_depth=i, min_samples_split=7)
    random_forest.fit(X_train, y_train)

    predictions = random_forest.predict(X_test)

    y_test_no_outliers = remove_outliers_iqr(y_test)

    predictions = predictions[:len(y_test_no_outliers)]

    mse_no_outliers = np.mean((predictions - y_test_no_outliers) ** 2)

    mse_usd = rubles_to_usd(mse_no_outliers)
    print("Mean Squared Error (MSE) without outliers in USD:", mse_usd)

    mae_no_outliers = np.mean(np.abs(predictions - y_test_no_outliers))

    mae_usd = rubles_to_usd(mae_no_outliers)
    print("Mean Absolute Error (MAE) without outliers in USD:", mae_usd)

    rmse_no_outliers = np.sqrt(mse_no_outliers)

    rmse_usd = rubles_to_usd(rmse_no_outliers)
    print("Root Mean Squared Error (RMSE) without outliers in USD:", rmse_usd)

    avg_percent_error_no_outliers = np.mean(np.abs((predictions - y_test_no_outliers) / y_test_no_outliers)) * 100
    print("Average Percentage Error without outliers:", avg_percent_error_no_outliers, "%")

#graphing
# results_df = pd.DataFrame(results)

# results_df.to_csv('results3.csv', index=False)

# plt.plot(results_df['i'], results_df['Avg_percent_error'], marker='o')
# plt.title('Depth vs Average Percentage Error')
# plt.xlabel('Depth')
# plt.ylabel('Average Percentage Error (%)')
# plt.grid(True)
# plt.show()
