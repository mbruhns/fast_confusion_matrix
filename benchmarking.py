import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
from fast_confusion_matrix.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    jaccard_score,
)

from sklearn.metrics import precision_score as precision_score_sklearn
from sklearn.metrics import recall_score as recall_score_sklearn
from sklearn.metrics import accuracy_score as accuracy_score_sklearn
from sklearn.metrics import f1_score as f1_score_sklearn
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy_score_sklearn
from sklearn.metrics import matthews_corrcoef as matthews_corrcoef_sklearn
from sklearn.metrics import jaccard_score as jaccard_score_sklearn

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
    parser = argparse.ArgumentParser(
        description="Arguments for input size and number of iterations."
    )

    # Add arguments
    parser.add_argument(
        "--input_size", type=int, default=10_000, help="Size of input (default: 50_000)"
    )
    parser.add_argument(
        "--n_iter", type=int, default=50, help="Number of iterations (default: 50)"
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    n_iter = 15  # args.n_iter

    # Access the values
    # input_size = 500_000  #args.input_size

    function_lst = [
        (precision_score, precision_score_sklearn),
        (recall_score, recall_score_sklearn),
        (accuracy_score, accuracy_score_sklearn),
        (f1_score, f1_score_sklearn),
        (balanced_accuracy_score, balanced_accuracy_score_sklearn),
        (matthews_corrcoef, matthews_corrcoef_sklearn),
        (jaccard_score, jaccard_score_sklearn),
    ]

    dct_lst = []
    # input_size_lst = [10, 100, 1000, 10000, 100000, 1000000]
    input_size_lst = np.logspace(1, 7, 15).astype(int)

    for input_size in input_size_lst:
        print(f"Compare runtimes for {input_size:,} elements and {n_iter} iterations.")
        speedup_dict = {}

        for nb_func, sk_func in function_lst:
            nb_time, sk_time = runtime_calculation(
                nb_func=nb_func, sk_func=sk_func, input_size=input_size, n_iter=n_iter
            )
            speedup = sk_time / nb_time
            speedup_dict[nb_func.py_func.__name__] = np.round(speedup, 2)

        dct_lst.append(speedup_dict)

    df = pd.DataFrame(dct_lst)
    df["input_size"] = input_size_lst

    print("Speedup results:")

    # Melt the DataFrame to long format
    df_melted = pd.melt(
        df, id_vars="input_size", var_name="metric", value_name="speedup"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot the data
    sns.lineplot(x="input_size", y="speedup", hue="metric", data=df_melted)
    plt.axhline(1, color="black", linestyle="--", label="baseline", zorder=-1)

    plt.legend(bbox_to_anchor=(1.1, 1.02))

    plt.title(
        "Speedup of numba implementation over sklearn implementation", fontsize=16
    )
    plt.xlabel("Input size", fontsize=14)
    plt.ylabel("Speedup over sklearn implementation", fontsize=14)

    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("speedup.png", dpi=300, bbox_inches="tight", transparent=False)
    plt.show()


if __name__ == "__main__":
    main()
