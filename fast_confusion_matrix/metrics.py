import numpy as np
from numba import njit


@njit(parallel=True, cache=True, fastmath=True)
def calculate_confusion_matrix(labels, predictions):
    # Some basing asserts
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    assert labels.ndim == predictions.ndim == 1

    tp = np.sum(labels * predictions)
    fn = np.sum(labels * (1 - predictions))
    fp = np.sum((1 - labels) * predictions)
    tn = np.sum((1 - labels) * (1 - predictions))

    return tp, tn, fp, fn


@njit(cache=True, fastmath=True)
def precision_score(y_true, y_pred):
    tp, tn, fp, fn = calculate_confusion_matrix(labels=y_true, predictions=y_pred)
    return tp / (tp + fp)


@njit(cache=True, fastmath=True)
def recall_score(y_true, y_pred):
    tp, tn, fp, fn = calculate_confusion_matrix(labels=y_true, predictions=y_pred)
    return tp / (tp + fn)


@njit(cache=True, fastmath=True)
def specificity_score(y_true, y_pred):
    tp, tn, fp, fn = calculate_confusion_matrix(labels=y_true, predictions=y_pred)
    return tn / (tn + fp)


@njit(cache=True, fastmath=True)
def accuracy(y_true, y_pred):
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total


@njit(cache=True, fastmath=True)
def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score (y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec)


@njit(cache=True, fastmath=True)
def balanced_accuracy_score(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    return (recall + specificity) / 2


@njit(cache=True, fastmath=True)
def matthews_corrcoef(y_true, y_pred):
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator


@njit(cache=True, fastmath=True)
def jaccard_score(y_true, y_pred):
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    return tp / (tp + fn + fp)
