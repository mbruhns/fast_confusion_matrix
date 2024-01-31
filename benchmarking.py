import pandas as pd
import seaborn as sns
import numpy as np
from time import perf_counter
from fast_confusion_matrix.metrics import (precision_score, recall_score, accuracy_score, f1_score,
                                           balanced_accuracy_score, matthews_corrcoef, jaccard_score)

from sklearn.metrics import precision_score as precision_score_sklearn
from sklearn.metrics import recall_score as recall_score_sklearn
from sklearn.metrics import accuracy_score as accuracy_score_sklearn
from sklearn.metrics import f1_score as f1_score_sklearn
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy_score_sklearn
from sklearn.metrics import matthews_corrcoef as matthews_corrcoef_sklearn
from sklearn.metrics import jaccard_score as jaccard_score_sklearn

from pprint import pprint
import argparse


def runtime_calculation(nb_func, sk_func, input_size=1_000, n_iter=50):
    # Generate random input data
    rng = np.random.default_rng()
    y_true = rng.integers(2, size=input_size)
    y_pred = rng.integers(2, size=input_size)

    start_time_nb = perf_counter()
    for _ in range(n_iter):
        _ = nb_func(y_true, y_pred)
    total_time_nb = (perf_counter() - start_time_nb) / n_iter

    start_time_sk = perf_counter()
    for _ in range(n_iter):
        _ = sk_func(y_true, y_pred)
    total_time_sk = (perf_counter() - start_time_sk) / n_iter

    return total_time_nb, total_time_sk


def main():
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description="Arguments for input size and number of iterations.")

    # Add arguments
    parser.add_argument("--input_size", type=int, default=10_000, help="Size of input (default: 50_000)")
    parser.add_argument("--n_iter", type=int, default=50, help="Number of iterations (default: 50)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values
    input_size = args.input_size
    n_iter = args.n_iter

    function_lst = [(precision_score, precision_score_sklearn),
                    (recall_score, recall_score_sklearn),
                    (accuracy_score, accuracy_score_sklearn),
                    (f1_score, f1_score_sklearn),
                    (balanced_accuracy_score, balanced_accuracy_score_sklearn),
                    (matthews_corrcoef, matthews_corrcoef_sklearn),
                    (jaccard_score, jaccard_score_sklearn)]

    print(f"Compare runtimes for {input_size:,} elements and {n_iter} iterations.")
    speedup_dict = {}

    for nb_func, sk_func in function_lst:
        nb_time, sk_time = runtime_calculation(nb_func=nb_func, sk_func=sk_func,  input_size=input_size, n_iter=n_iter)
        speedup = sk_time / nb_time
        speedup_dict[nb_func.py_func.__name__] = np.round(speedup,2)

    print("Speedup results:")
    pprint(speedup_dict)

if __name__ == "__main__":
    main()