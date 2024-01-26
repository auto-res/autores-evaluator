# F1-score
# accuracy
# logloss
# ROC-AUC
# PR-AUC
# pAUC
# Precission
# Recall
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, log_loss
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def binary_classification(y_test, y_prob):
    threshold = 0.5
    y_pred = np.where(np.array(y_prob) > threshold, 1, 0)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc_partial = roc_auc_score(y_test, y_prob, max_fpr=0.1)
    return
