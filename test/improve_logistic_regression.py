import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def feature_normalize(X):
    """特徴量の正規化"""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def compute_cost_reg(X, y, weights, lambda_):
    """正則化されたコスト関数"""
    m = len(y)
    h = sigmoid(X @ weights)
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + (lambda_ / (2*m)) * np.sum(weights[1:] ** 2)
    return cost

def gradient_descent_reg(X, y, weights, learning_rate, iterations, lambda_):
    """L2正則化を含む勾配降下法"""
    m = len(y)
    cost_history = []
    for i in range(iterations):
        h = sigmoid(X @ weights)
        gradient = (X.T @ (h - y)) / m
        gradient[1:] += (lambda_ / m) * weights[1:]  # バイアス項は正則化しない
        weights -= learning_rate * gradient
        cost = compute_cost_reg(X, y, weights, lambda_)
        cost_history.append(cost)

        # 早期停止の条件（改善が見られない場合）
        if i > 0 and abs(cost_history[-2] - cost_history[-1]) < 1e-6:
            print(f"Early stopping at iteration {i}")
            break

    return weights, cost_history

def predict(X, weights):
    predictions = sigmoid(X @ weights) >= 0.5
    return predictions.astype(int)

def model(X_train, y_train, X_valid, params):
    # 特徴量の正規化
    X_train_norm, mu, sigma = feature_normalize(X_train)
    X_valid_norm = (X_valid - mu) / sigma  # 検証データも同じパラメータで正規化

    # 訓練データにバイアス項を追加
    X_train_norm = np.insert(X_train_norm, 0, 1, axis=1)
    X_valid_norm = np.insert(X_valid_norm, 0, 1, axis=1)

    # 重みの初期化
    weights = np.zeros(X_train_norm.shape[1])

    # モデルの学習
    lambda_ = params["lambda"]  # 正則化パラメータ
    weights, cost_history = gradient_descent_reg(X_train_norm, y_train, weights, params["learning_rate"], params["iterations"], lambda_)

    # 検証データセットに対する予測
    y_pred = predict(X_valid_norm, weights)

    return y_pred


