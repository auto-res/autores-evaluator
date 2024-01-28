import lightgbm as lgb
from sklearn.model_selection import train_test_split


def model(X_train, y_train, X_valid, params):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    gbm = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

    y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    return y_pred
