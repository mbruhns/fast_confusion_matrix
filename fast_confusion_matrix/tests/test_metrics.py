import pytest
import numpy as np

from sklearn.metrics import precision_score as precision_score_sklearn
from sklearn.metrics import recall_score as recall_score_sklearn
from sklearn.metrics import accuracy_score as accuracy_score_sklearn
from sklearn.metrics import f1_score as f1_score_sklearn
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy_score_sklearn
from sklearn.metrics import matthews_corrcoef as matthews_corrcoef_sklearn
from sklearn.metrics import jaccard_score as jaccard_score_sklearn

from fast_confusion_matrix.metrics import (precision_score, recall_score, accuracy, f1_score, balanced_accuracy_score,
                                           matthews_corrcoef, jaccard_score)

function_lst = [(precision_score, precision_score_sklearn),
                (recall_score, recall_score_sklearn),
                (accuracy, accuracy_score_sklearn),
                (f1_score, f1_score_sklearn),
                (balanced_accuracy_score, balanced_accuracy_score_sklearn),
                (matthews_corrcoef, matthews_corrcoef_sklearn),
                (jaccard_score, jaccard_score_sklearn)]

@pytest.mark.parametrize("nb_func, sk_func", function_lst)
def test_check_func_equal(nb_func, sk_func, input_size=100):
    # Generate random input data
    rng = np.random.default_rng()
    y_true = rng.integers(2, size=input_size)
    y_pred = rng.integers(2, size=input_size)

    # Calculate results using both functions
    nb_res = nb_func(y_true, y_pred)
    sk_res = sk_func(y_true, y_pred)

    # Assert that the results are equal
    np.testing.assert_array_equal(nb_res,
                                  sk_res,
                                  err_msg=f"{nb_func.py_func.__name__} is not correctly implemented.")
