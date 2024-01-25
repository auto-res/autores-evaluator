import lightgbm as lgb
from sklearn.model_selection import train_test_split



def model(X_train, y_train, X_valid):
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # パラメータ設定
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1
    }

    gbm = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, early_stopping_rounds=10)

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    return y_pred
