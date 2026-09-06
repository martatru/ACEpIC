import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

DATASET_PATH = "embeddings_data/cleaned_peptides.csv"
EMBEDDINGS_DIR = "embeddings_data"
OUTPUT_PLOT_PATH = "scripts/visualization/plots/ablation_heatmap_matrix.png"

EMBEDDING_FILES = {
    "ESM-2 (650M)": "esm2_650m_embeddings.npy",
    "ESM-2 (35M)": "esm2_35m_embeddings.npy",
    "ESM-2 (8M)": "esm2_8m_embeddings.npy",
    "ProtT5-XL": "prott5_embeddings.npy",
    "Ankh Base": "ankh_base_embeddings.npy",
}


def load_dataset(dataset_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load target pIC50 values and dataset dataframe."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)
    if "pIC50" not in df.columns:
        raise KeyError("Column 'pIC50' was not found in dataset CSV.")

    return df, df["pIC50"].values


def get_regression_models() -> Dict[str, object]:
    """Return machine learning regressors for ablation benchmarking."""
    return {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "SVR": SVR(C=1.0, kernel="rbf"),
    }


def identify_noisy_samples(
    X: np.ndarray, y: np.ndarray, noise_percentile: float = 15.0
) -> np.ndarray:
    """Filter noise using out-of-fold cross-validation prediction residuals."""
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    residuals = np.zeros_like(y, dtype=np.float64)

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        residuals[val_idx] = np.abs(y_val - preds)

    threshold = np.percentile(residuals, 100.0 - noise_percentile)
    return residuals <= threshold


def evaluate_pipeline_matrix(
    y: np.ndarray, clean_mask: np.ndarray, n_splits: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate 25 model-embedding pairs on raw and cleaned datasets."""
    models = get_regression_models()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    raw_matrix = pd.DataFrame(
        index=list(models.keys()),
        columns=list(EMBEDDING_FILES.keys()),
        dtype=float,
    )
    clean_matrix = pd.DataFrame(
        index=list(models.keys()),
        columns=list(EMBEDDING_FILES.keys()),
        dtype=float,
    )

    for emb_name, filename in EMBEDDING_FILES.items():
        filepath = os.path.join(EMBEDDINGS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {emb_name}: File {filepath} not found.")
            continue

        X_raw = np.load(filepath)
        X_clean = X_raw[clean_mask]
        y_clean = y[clean_mask]

        print(f"Training models on embedding: {emb_name}...")

        for model_name, model_obj in models.items():
            # Raw dataset cross-validation
            raw_pearsons = []
            for train_idx, val_idx in kf.split(X_raw, y):
                model_obj.fit(X_raw[train_idx], y[train_idx])
                preds = model_obj.predict(X_raw[val_idx])
                r_val, _ = pearsonr(y[val_idx], preds)
                raw_pearsons.append(r_val)
            raw_matrix.loc[model_name, emb_name] = float(np.mean(raw_pearsons))

            # Cleaned dataset cross-validation
            clean_pearsons = []
            for train_idx, val_idx in kf.split(X_clean, y_clean):
                model_obj.fit(X_clean[train_idx], y_clean[train_idx])
                preds = model_obj.predict(X_clean[val_idx])
                r_val, _ = pearsonr(y_clean[val_idx], preds)
                clean_pearsons.append(r_val)
            clean_matrix.loc[model_name, emb_name] = float(
                np.mean(clean_pearsons)
            )

    return raw_matrix, clean_matrix


def plot_ablation_heatmaps(
    raw_df: pd.DataFrame, clean_df: pd.DataFrame, output_path: str
) -> None:
    """Generate side-by-side heatmaps for ablation benchmark comparison."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="white", font="sans-serif")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)

    vmin = min(raw_df.values.min(), clean_df.values.min())
    vmax = max(raw_df.values.max(), clean_df.values.max())

    cmap = sns.light_palette("#7c3aed", as_cmap=True)

    # Panel A: Raw Dataset Matrix
    ax1 = axes[0]
    sns.heatmap(
        raw_df,
        ax=ax1,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        linewidths=1,
        linecolor="#ffffff",
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax1.set_title(
        "Raw Dataset (Pearson r)", fontsize=13, fontweight="bold", pad=12
    )
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontweight="bold")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")

    # Panel B: Cleaned Dataset Matrix
    ax2 = axes[1]
    sns.heatmap(
        clean_df,
        ax=ax2,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Pearson Correlation (r)"},
        linewidths=1,
        linecolor="#ffffff",
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax2.set_title(
        "Cleaned Dataset (Pearson r)", fontsize=13, fontweight="bold", pad=12
    )
    ax2.set_yticklabels([], rotation=0)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")

    plt.suptitle(
        "Ablation Study: Embedding & Model Performance Across Data States",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Ablation heatmap matrix saved to: {output_path}")


def main() -> None:
    """Execute evaluation pipeline and generate side-by-side heatmaps."""
    try:
        df, y = load_dataset(DATASET_PATH)

        reference_file = os.path.join(
            EMBEDDINGS_DIR, EMBEDDING_FILES["ESM-2 (650M)"]
        )
        X_ref = np.load(reference_file)
        clean_mask = identify_noisy_samples(X_ref, y, noise_percentile=15.0)

        raw_df, clean_df = evaluate_pipeline_matrix(y, clean_mask)
        plot_ablation_heatmaps(raw_df, clean_df, OUTPUT_PLOT_PATH)

    except Exception as error:
        print(f"Pipeline execution failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()