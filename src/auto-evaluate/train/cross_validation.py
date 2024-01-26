from sklearn.model_selection import KFold

def cross_validataion(model, dataset, metrix):
    X = dataset.drop(columns=['survived']).values
    y = dataset['survived'].values

    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_pred = model(X_train, y_train, X_test)
        metrix(y_test, y_pred)
    return
