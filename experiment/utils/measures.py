import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


minClass = 1  # label of minority class
majClass = 0  # label of majority class


def harmonic_mean(x, y, beta=1):
    beta *= beta
    return (beta + 1) * x * y / np.array(beta * x + y)


def get_metrics(Ytest, Ytest_pred):
    """
    Compute performance measures by comparing prediction with true labels
    :param Ytest: real label
    :param Ytest_pred:  predict label
    :return:
    """
    TN, FP, FN, TP = confusion_matrix(Ytest, Ytest_pred,
                                      labels=[majClass, minClass]).ravel()
    return TN, FP, FN, TP


def Accuracy(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    accuracy = (TN + TP) / (TN + FP + FN + TP)
    return accuracy


def Precision(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    precision = TP / (TP + FP)
    return precision


def Recall(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    recall = TP / (TP + FN)
    return recall


def F1(Ytest, Ytest_pred):
    # perf = {}
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    # perf['test'] = ((int(tn), int(fp), int(fn), int(tp)))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = harmonic_mean(precision, recall)
    return F1


def F_negative(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    F_negative = harmonic_mean(TN / (TN + FN), TN / (TN + FP))
    return F_negative


def MCC(Ytest, Ytest_pred):
    TN, FP, FN, TP = get_metrics(Ytest, Ytest_pred)
    mcc = np.array([TP + FN, TP + FP, FN + TN, FP + TN]).prod()
    MCC = (TP * TN - FN * FP) / np.sqrt(mcc)
    return MCC


def AUC(Ytest, Ytest_pred):
    auc = roc_auc_score(Ytest, Ytest_pred)
    return auc


