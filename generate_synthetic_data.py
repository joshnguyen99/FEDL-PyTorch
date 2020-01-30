import numpy as np
import json
import random

np.random.seed(0)


def generate_synthetic_non_iid(alpha, beta, p=1.2, NUM_USERS=100,
                               DIMENSION=60, NUM_CLASSES=10, kappa=10):
    """
    Generate non-iid synthetic data for classification.
    :return: X and y for each user
    :param alpha: stddev of u
    :param beta: stddev of B
    :param p: parameter in diagonal matrix (controlling kappa)
    :param NUM_USERS: total number of users
    :param DIMENSION: dimension of data points
    :param NUM_CLASSES: number of output classes
    :return:
    """

    # For consistent outcomes
    np.random.seed(0)

    OUTPUT_DIM = 1 if NUM_CLASSES == 2 else NUM_CLASSES  # Determine if logistic regression
    SAMPLES_PER_USER = np.random.lognormal(4, 2, NUM_USERS).astype(int) + 50
    NUM_TOTAL_SAMPLES = np.sum(SAMPLES_PER_USER)

    X_split = [[] for _ in range(NUM_USERS)]   # X for each user
    y_split = [[] for _ in range(NUM_USERS)]   # y for each user

    u = np.random.normal(0, alpha, NUM_USERS)  # u_k = mean of W_k
    B = np.random.normal(0, beta, NUM_USERS)   # B_k = mean of v_k

    # Find v (mean of X)
    v = np.zeros((NUM_USERS, DIMENSION))
    for k in range(NUM_USERS):
        v[k] = np.random.normal(B[k], 1, DIMENSION)

    # Covariance matrix
    diagonal = np.array([(j+1) ** -p for j in range(DIMENSION)])
    Sigma = np.diag(diagonal)

    def softmax(x):
        exp = np.exp(x)
        return exp / np.sum(exp)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Generate X for each user
    LAMBDA = 100 if kappa == 1 else (1 / (kappa - 1))
    max_norm = 0
    for k in range(NUM_USERS):
        X_k = np.random.multivariate_normal(v[k], Sigma, SAMPLES_PER_USER[k])
        max_norm = max(max_norm,
                       np.linalg.norm(X_k.T.dot(X_k), 2) / SAMPLES_PER_USER[k])
        X_split[k] = X_k
    # Normalize X for each user using max_norm and LAMBDA
    for k in range(NUM_USERS):
        X_split[k] /= max_norm + LAMBDA

    # Generate y for each user
    for k in range(NUM_USERS):
        # W_k ~ N(u_k, 1) (In Network-DANE W_k is generated uniformly randomly)
        W_k = np.random.normal(u[k], 1, (DIMENSION, OUTPUT_DIM))

        # b_k ~ N(u_k, 1) (In Network-DANE there is no bias)
        b_k = np.random.normal(u[k], 1, OUTPUT_DIM)

        X_k = X_split[k]

        y_k = np.zeros(SAMPLES_PER_USER[k])
        for i in range(SAMPLES_PER_USER[k]):
            if NUM_CLASSES == 2:
                # Logistic regression
                # (In Network-DANE y_k = sigmoid(W_k * x_k + noise)
                # where noise ~ N(0, I))
                y_k[i] = int(sigmoid(np.dot(X_k[i], W_k) + b_k) > 0.5)
            else:
                # Multinomial regression
                y_k[i] = np.argmax(softmax(np.dot(X_k[i], W_k) + b_k))

        X_split[k] = X_split[k].tolist()
        y_split[k] = y_k.tolist()

        print("User {} has {} data points.".format(k, y_k.shape[0]))

    print("Total number of samples: {}".format(NUM_TOTAL_SAMPLES))
    return X_split, y_split


def main():
    NUM_USERS = 100

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    train_path = "data/train/mytrain.json"
    test_path = "data/test/mytest.json"

    X, y = generate_synthetic_non_iid(alpha=0, beta=0)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in range(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.75 * num_samples)
        test_len = num_samples - train_len

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == '__main__':
    main()
