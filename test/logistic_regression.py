import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(X.dot(weights))
    cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


def gradient_descent(X, y, weights, learning_rate, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X.dot(weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= learning_rate * gradient
        cost = compute_cost(X, y, weights)
        cost_history.append(cost)

    return weights, cost_history


def predict(X, weights):
    predictions = sigmoid(X.dot(weights)) >= 0.5
    return predictions.astype(int)


def model(X_train, y_train, X_valid, params):
    # 訓練データにバイアス項を追加
    X_train = np.insert(X_train, 0, 1, axis=1)
    X_valid = np.insert(X_valid, 0, 1, axis=1)

    # 重みの初期化
    weights = np.zeros(X_train.shape[1])

    # モデルの学習
    weights, cost_history = gradient_descent(
        X_train, y_train, weights, params["learning_rate"], params["iterations"]
    )

    # 検証データセットに対する予測
    y_pred = predict(X_valid, weights)

    return y_pred
