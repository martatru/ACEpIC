import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

EMBEDDINGS_PATH = "embeddings_data/esm2_650m_embeddings.npy"
DATASET_PATH = "embeddings_data/cleaned_peptides.csv"


def load_dataset_and_embeddings(
    embeddings_path: str, dataset_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature embeddings matrix and target pIC50 values.

    Args:
        embeddings_path: Path to the .npy file containing embeddings.
        dataset_path: Path to the CSV file containing target pIC50 labels.

    Returns:
        A tuple containing the features matrix X and target array y.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    X = np.load(embeddings_path)
    df = pd.read_csv(dataset_path)

    if "pIC50" not in df.columns:
        raise KeyError("Column 'pIC50' was not found in the dataset CSV.")

    y = df["pIC50"].values
    return X, y


def evaluate_ridge_baseline(
    X: np.ndarray, y: np.ndarray, n_splits: int = 10, random_state: int = 42
) -> Dict[str, list]:
    """Perform k-fold cross-validation using Ridge Regression.

    Args:
        X: Feature matrix of embeddings.
        y: Target pIC50 continuous values.
        n_splits: Number of cross-validation folds.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary containing metric names mapped to lists of fold scores.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics = {
        "rmse": [],
        "mae": [],
        "r2": [],
        "pearson": [],
        "spearman": [],
    }

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse_val = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_val = float(mean_absolute_error(y_val, y_pred))
        r2_val = float(r2_score(y_val, y_pred))
        p_corr, _ = pearsonr(y_val, y_pred)
        s_corr, _ = spearmanr(y_val, y_pred)

        metrics["rmse"].append(rmse_val)
        metrics["mae"].append(mae_val)
        metrics["r2"].append(r2_val)
        metrics["pearson"].append(p_corr)
        metrics["spearman"].append(s_corr)

    return metrics


def print_evaluation_summary(metrics: Dict[str, list]) -> None:
    """Print averaged metrics with standard deviation."""
    print("=== BASELINE EVALUATION SUMMARY ===")
    print("Model: Ridge Regression (alpha=1.0)")
    print("Features: ESM-2 650M Embeddings")
    print("Validation: 10-Fold Cross-Validation")
    print("-----------------------------------")
    for metric_name, score_list in metrics.items():
        mean_score = np.mean(score_list)
        std_score = np.std(score_list)
        print(f"{metric_name.upper():<10}: {mean_score:.4f} +/- {std_score:.4f}")


def main() -> None:
    """Main execution function for training the baseline model."""
    try:
        X, y = load_dataset_and_embeddings(EMBEDDINGS_PATH, DATASET_PATH)
        print(f"Dataset loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
        metrics = evaluate_ridge_baseline(X, y, n_splits=10)
        print_evaluation_summary(metrics)
    except Exception as error:
        print(f"Execution failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()