import os
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

DATASET_PATH = "embeddings_data/cleaned_peptides.csv"
EMBEDDINGS_PATH = "embeddings_data/esm2_650m_embeddings.npy"
OUTPUT_PLOT_PATH = "scripts/visualization/plots/ml_models_comparison.png"


def load_dataset_and_embeddings(
    embeddings_path: str, dataset_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ESM-2 embeddings matrix and target pIC50 continuous values."""
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found at: {embeddings_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    X = np.load(embeddings_path)
    df = pd.read_csv(dataset_path)

    if "pIC50" not in df.columns:
        raise KeyError("Column 'pIC50' was not found in dataset CSV.")

    y = df["pIC50"].values
    return X, y


def get_regression_models() -> Dict[str, object]:
    """Return dictionary of machine learning regressors inspired by pLM4ACE."""
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Support Vector (SVR)": SVR(C=1.0, kernel="rbf"),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "K-Neighbors": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        ),
    }


def evaluate_models_cross_validation(
    X: np.ndarray, y: np.ndarray, n_splits: int = 10, random_state: int = 42
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Evaluate machine learning regressors using k-fold cross-validation."""
    models = get_regression_models()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    summary_metrics = {}
    oof_predictions = {}

    for model_name, model_obj in models.items():
        print(f"Evaluating {model_name}...")
        y_pred_oof = np.zeros_like(y, dtype=np.float64)

        fold_pearsons = []
        fold_spearmans = []
        fold_rmses = []
        fold_r2s = []

        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model_obj.fit(X_train, y_train)
            preds = model_obj.predict(X_val)

            y_pred_oof[val_idx] = preds
            p_corr, _ = pearsonr(y_val, preds)
            s_corr, _ = spearmanr(y_val, preds)

            fold_pearsons.append(p_corr)
            fold_spearmans.append(s_corr)
            fold_rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
            fold_r2s.append(r2_score(y_val, preds))

        summary_metrics[model_name] = {
            "pearson_mean": float(np.mean(fold_pearsons)),
            "pearson_std": float(np.std(fold_pearsons)),
            "spearman_mean": float(np.mean(fold_spearmans)),
            "spearman_std": float(np.std(fold_spearmans)),
            "rmse_mean": float(np.mean(fold_rmses)),
            "r2_mean": float(np.mean(fold_r2s)),
        }
        oof_predictions[model_name] = y_pred_oof

    return summary_metrics, oof_predictions


def create_ml_comparison_plot(
    y_true: np.ndarray,
    metrics_summary: Dict[str, Dict[str, float]],
    oof_predictions: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Generate a clean pastel benchmark plot comparing ML regressors."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="white", font="sans-serif")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Determine top performing model based on Pearson r
    best_model_name = max(
        metrics_summary.keys(),
        key=lambda k: metrics_summary[k]["pearson_mean"],
    )

    y_pred = oof_predictions[best_model_name]
    p_mean = metrics_summary[best_model_name]["pearson_mean"]
    s_mean = metrics_summary[best_model_name]["spearman_mean"]
    rmse_mean = metrics_summary[best_model_name]["rmse_mean"]
    r2_mean = metrics_summary[best_model_name]["r2_mean"]

    # -------------------------------------------------------------------------
    # Panel A: Scatter Plot & Regression Line (Top ML Model)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.grid(False)

    sns.regplot(
        x=y_true,
        y=y_pred,
        ax=ax1,
        scatter_kws={"alpha": 0.30, "s": 16, "color": "#b19ffb"},
        line_kws={"color": "#7c3aed", "linewidth": 2, "label": "Regression Fit"},
    )

    min_val = min(y_true.min(), y_pred.min()) - 0.2
    max_val = max(y_true.max(), y_pred.max()) + 0.2

    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.set_aspect("equal", adjustable="box")

    ax1.set_xlabel("Experimental pIC50", fontsize=11, labelpad=8)
    ax1.set_ylabel("Predicted pIC50 (10-CV OOF)", fontsize=11, labelpad=8)
    ax1.set_title(
        f"Actual vs Predicted pIC50 [{best_model_name}]",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    ax1.legend(loc="upper left", frameon=True, facecolor="#ffffff", framealpha=0.9)

    metrics_text = (
        f"Pearson r : {p_mean:.4f}\n"
        f"Spearman ρ: {s_mean:.4f}\n"
        f"R² Score  : {r2_mean:.4f}\n"
        f"RMSE      : {rmse_mean:.4f}"
    )
    ax1.text(
        0.60,
        0.06,
        metrics_text,
        transform=ax1.transAxes,
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#f5f3ff",
            alpha=0.9,
            edgecolor="#ddd6fe",
        ),
    )

    # -------------------------------------------------------------------------
    # Panel B: Pastel Horizontal Barplot (ML Model Comparison)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.grid(False)

    df_bar = pd.DataFrame(
        [
            {
                "Model": lbl,
                "Pearson r": stats["pearson_mean"],
                "std": stats["pearson_std"],
            }
            for lbl, stats in metrics_summary.items()
        ]
    )

    df_bar = df_bar.sort_values(by="Pearson r", ascending=False).reset_index(
        drop=True
    )

    sns.barplot(
        data=df_bar,
        y="Model",
        x="Pearson r",
        ax=ax2,
        palette="pastel",
        alpha=0.9,
        edgecolor="#64748b",
        linewidth=0.8,
    )

    for i, row in df_bar.iterrows():
        r_val = row["Pearson r"]
        std_val = row["std"]

        ax2.errorbar(
            x=r_val,
            y=i,
            xerr=std_val,
            fmt="none",
            ecolor="#475569",
            capsize=3,
            linewidth=1.2,
        )

        label_x_pos = r_val + std_val + 0.015
        ax2.text(
            label_x_pos,
            i,
            f"r = {r_val:.3f}",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#334155",
        )

    max_x_needed = (df_bar["Pearson r"] + df_bar["std"]).max() + 0.12
    ax2.set_xlim(0, max_x_needed)

    ax2.set_xlabel("Pearson Correlation (r)", fontsize=11, labelpad=8)
    ax2.set_ylabel("", fontsize=11)
    ax2.set_title(
        "ML Algorithm Comparison on ESM-2 650M",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    sns.despine(ax=ax1, top=True, right=True)
    sns.despine(ax=ax2, top=True, right=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"ML comparison plot successfully saved to: {output_path}")


def main() -> None:
    """Execution pipeline for benchmarking ML models."""
    try:
        X, y = load_dataset_and_embeddings(EMBEDDINGS_PATH, DATASET_PATH)
        print(
            f"Dataset loaded successfully. Features shape: {X.shape}, Target shape: {y.shape}"
        )

        metrics_summary, oof_predictions = evaluate_models_cross_validation(X, y)
        create_ml_comparison_plot(
            y, metrics_summary, oof_predictions, OUTPUT_PLOT_PATH
        )
    except Exception as error:
        print(f"Pipeline failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()