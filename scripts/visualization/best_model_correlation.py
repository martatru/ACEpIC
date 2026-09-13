import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# Resolve repository root directory dynamically (ACEpIC/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    """Paths and settings for best model correlation visualization."""

    DATASET_PATH: Path = PROJECT_ROOT / "embeddings_data" / "cleaned_peptides.csv"
    PROTT5_PATH: Path = PROJECT_ROOT / "embeddings_data" / "prott5_embeddings.npy"
    ESM2_650M_PATH: Path = (
        PROJECT_ROOT / "embeddings_data" / "esm2_650m_embeddings.npy"
    )
    OUTPUT_PLOT_PATH: Path = (
        PROJECT_ROOT
        / "scripts"
        / "visualization"
        / "plots"
        / "best_model_correlation.png"
    )
    OUTPUT_PDF_PATH: Path = (
        PROJECT_ROOT
        / "scripts"
        / "visualization"
        / "plots"
        / "best_model_correlation.pdf"
    )
    RIDGE_ALPHA: float = 1.0
    CUTOFF_PERCENTILE: float = 80.0  # Retain 80% (remove 20% highest residuals)
    N_SPLITS: int = 10
    RANDOM_SEED: int = 42


def load_fused_dataset(
    config: Config = Config(),
) -> Tuple[np.ndarray, np.ndarray]:
    """Load targets and combine ProtT5 + ESM-2 650M representations."""
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV missing at: {config.DATASET_PATH}")
    if not config.PROTT5_PATH.exists() or not config.ESM2_650M_PATH.exists():
        raise FileNotFoundError("Embedding .npy files missing in embeddings_data/")

    df = pd.read_csv(config.DATASET_PATH)
    if "pIC50" not in df.columns:
        raise KeyError("Column 'pIC50' missing from dataset CSV.")

    y = df["pIC50"].values
    x_prott5 = np.load(config.PROTT5_PATH)
    x_esm2 = np.load(config.ESM2_650M_PATH)
    X_fused = np.hstack((x_prott5, x_esm2))

    return X_fused, y


def get_residual_cleaned_data(
    X: np.ndarray, y: np.ndarray, cutoff_percentile: float = 80.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter 20% noisy samples based on out-of-fold prediction residuals."""
    kf = KFold(n_splits=10, shuffle=True, random_state=Config.RANDOM_SEED)
    residuals = np.zeros_like(y, dtype=np.float64)

    for train_idx, val_idx in kf.split(X, y):
        model = Ridge(alpha=Config.RIDGE_ALPHA)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        residuals[val_idx] = np.abs(y[val_idx] - preds)

    threshold = np.percentile(residuals, cutoff_percentile)
    clean_mask = residuals <= threshold
    return X[clean_mask], y[clean_mask]


def generate_oof_predictions(
    X: np.ndarray, y: np.ndarray, n_splits: int = 10
) -> np.ndarray:
    """Generate 10-Fold Out-Of-Fold predictions for evaluation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=Config.RANDOM_SEED)
    y_pred_oof = np.zeros_like(y, dtype=np.float64)

    for train_idx, val_idx in kf.split(X, y):
        model = Ridge(alpha=Config.RIDGE_ALPHA)
        model.fit(X[train_idx], y[train_idx])
        y_pred_oof[val_idx] = model.predict(X[val_idx])

    return y_pred_oof


def plot_single_correlation(
    y_true: np.ndarray, y_pred: np.ndarray, config: Config
) -> None:
    """Generate single-panel correlation plot with title and matcha green palette."""
    config.OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    sns.set_theme(style="white", font="sans-serif")

    pearson_r, _ = pearsonr(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.grid(False)

    # Vivid Matcha Green Palette (#8eb486 / #3b5238)
    sns.regplot(
        x=y_true,
        y=y_pred,
        ax=ax,
        scatter_kws={"alpha": 0.50, "s": 20, "color": "#8eb486"},
        line_kws={"color": "#3b5238", "linewidth": 1.8, "label": "Regression fit"},
    )

    min_val = min(y_true.min(), y_pred.min()) - 0.2
    max_val = max(y_true.max(), y_pred.max()) + 0.2

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Experimental pIC50", fontsize=10, fontweight="bold", labelpad=6)
    ax.set_ylabel(
        "Predicted pIC50 (10-Fold CV OOF)", fontsize=10, fontweight="bold", labelpad=6
    )
    ax.tick_params(labelsize=9)

    # Title
    ax.set_title(
        "ACEpIC Best Model: Experimental vs. Predicted pIC50\n"
        "Fused (ProtT5-XL + ESM-2 650M) | Ridge | 20% Residual Data Cleaning",
        fontsize=10.5,
        fontweight="bold",
        pad=12,
    )

    ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    metrics_text = (
        f"Pearson r  : {pearson_r:.4f}\n"
        f"Spearman ρ : {spearman_rho:.4f}\n"
        f"R²         : {r2:.4f}\n"
        f"RMSE       : {rmse:.4f}\n"
        f"MAE        : {mae:.4f}"
    )
    ax.text(
        0.56,
        0.06,
        metrics_text,
        transform=ax.transAxes,
        fontsize=8.5,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f2f7f0",
            alpha=0.95,
            edgecolor="#c2d8bc",
            linewidth=0.8,
        ),
    )

    sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()
    plt.savefig(config.OUTPUT_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.savefig(config.OUTPUT_PDF_PATH, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Raster plot saved to: {config.OUTPUT_PLOT_PATH}")
    print(f"Vector PDF saved to: {config.OUTPUT_PDF_PATH}")


def main() -> None:
    """Execution entry point."""
    try:
        config = Config()
        X_fused, y_full = load_fused_dataset(config)
        X_clean, y_clean = get_residual_cleaned_data(
            X_fused, y_full, cutoff_percentile=config.CUTOFF_PERCENTILE
        )
        y_pred_oof = generate_oof_predictions(
            X_clean, y_clean, n_splits=config.N_SPLITS
        )
        plot_single_correlation(y_clean, y_pred_oof, config)

    except Exception as error:
        print(f"Execution failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()