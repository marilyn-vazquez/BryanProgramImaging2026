"""
Module Documentation: Batch Classification Experiment Pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


def run_batch_experiments(
        input_dir,
        output_dir,
        filtration_suffix="_dilation_barcode.npy",
        n_iterations=100):
    """
    Executes batch classification experiments using machine learning models 
    (Support Vector Machine and Multi-Layer Perceptron) on numerical feature 
    arrays (dilation barcodes). It performs repeated stratified train-test splits 
    over a specified number of iterations, evaluates performance using accuracy 
    and F1 scores, and saves both raw and summary statistical CSV reports to 
    a specified output directory.

    Parameters
    ----------
    input_dir : str or pathlib.Path
        Path to the directory containing the input feature files (NumPy .npy arrays).
    output_dir : str or pathlib.Path
        Path to the directory where resulting CSV files and summary reports will 
        be saved. Created automatically if it does not exist.
    filtration_suffix : str, optional
        File name suffix used to locate and filter target input files within 
        the input_dir. Default is "_dilation_barcode.npy".
    n_iterations : int, optional
        Number of randomized train-test split iterations (and distinct random seeds) 
        to run for cross-validation evaluation. Default is 100.

    Inputs
    ------
    - Files on Disk: NumPy binary files (.npy) located inside `input_dir` matching 
      the `filtration_suffix`.
    - Labeling Logic: Binary class labels are derived automatically from filenames:
        * 1: Assigned if the filename contains the substring "microgravity" (case-insensitive).
        * 0: Assigned to all other control files.

    Outputs
    -------
    - Console Logs: Prints runtime progress updates, dataset dimensions, and completion status.
    - CSV Files Saved to `output_dir`:
        1. [ClassifierName]_100_iteration_results.csv: Contains individual performance 
           metrics (Accuracy, F1) across all iterations.
        2. [ClassifierName]_summary_statistics.csv: Contains statistical summaries 
           (Mean, Std) computed across the iteration runs for each classifier.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(f"*{filtration_suffix}"))

    if not files:
        print(f"⚠️ No dilation barcode files found in {input_dir}")
        return

    X = []
    y = []

    for f in files:
        X.append(np.load(f))

        # Label:
        # 1 = microgravity
        # 0 = control
        y.append(1 if "microgravity" in f.name.lower() else 0)

    X = np.asarray(X)
    y = np.asarray(y)

    print(f"Loaded {len(X)} dilation barcode vectors.")
    print(f"Feature shape: {X.shape}")

    classifiers = {

        "Dilation_SVM": SVC(
            kernel="rbf",
            gamma=2
        ),

        "Dilation_NN": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=1000,
            random_state=42
        )
    }

    results = defaultdict(list)

    print(f"🚀 Running {n_iterations} dilation iterations...")

    for seed in range(n_iterations):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=seed,
            stratify=y
        )

        for name, clf in classifiers.items():

            pipe = make_pipeline(
                StandardScaler(),
                clf
            )

            pipe.fit(X_train, y_train)

            pred = pipe.predict(X_test)

            results[name].append({
                "Accuracy": accuracy_score(y_test, pred),
                "F1": f1_score(
                    y_test,
                    pred,
                    zero_division=0
                )
            })

    # Save results
    for name, data in results.items():

        df = pd.DataFrame(data)

        df.to_csv(
            output_dir /
            f"{name}_100_iteration_results.csv",
            index=False
        )

        summary = pd.DataFrame({
            "Mean": df.mean(),
            "Std": df.std()
        })

        summary.to_csv(
            output_dir /
            f"{name}_summary_statistics.csv"
        )

    print("\n✅ Dilation analysis complete.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":

    # Folder containing dilation barcode vectors
    BARCODE_DIR = Path(
        r"C:\Users\chloe.jamieson\OneDrive - Simpson College\Documents\GitHub\BryanProgramImaging2026\Experiments\IMAGES2.0\All Images\dilation_barcodes"
    )

    # Results folder
    RESULTS_DIR = BARCODE_DIR / "analysis_results"

    run_batch_experiments(
        BARCODE_DIR,
        RESULTS_DIR,
        filtration_suffix="_dilation_barcode.npy",
        n_iterations=100
    )
