import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Resolve repository root directory dynamically (ACEpIC/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    """Settings for square-layout summary ablation heatmap with compact, high-contrast labels."""

    RESULTS_CACHE_PATH: Path = (
        PROJECT_ROOT / "embeddings_data" / "master_ablation_results.csv"
    )
    OUTPUT_PLOT_PATH: Path = (
        PROJECT_ROOT
        / "scripts"
        / "visualization"
        / "plots"
        / "summary_ablation_heatmap.png"
    )
    OUTPUT_PDF_PATH: Path = (
        PROJECT_ROOT
        / "scripts"
        / "visualization"
        / "plots"
        / "summary_ablation_heatmap.pdf"
    )

    # Selected representative models
    SELECTED_MODELS: List[str] = [
        "ESM-2 (8M) - KNN",
        "ESM-2 (650M) - Random Forest",
        "ProtT5-XL - Ridge",
        "Fused (ProtT5 + ESM2) - XGBoost",
        "Fused (ProtT5 + ESM2) - Ridge",
    ]

    # Selected evaluation & cleaning scenarios
    SELECTED_SCENARIOS: List[str] = [
        "10-Fold CV\n[Raw Data]",
        "10-Fold CV\n[Residual OOF (20%)]",
        "Cluster Split (80/20)\n[Raw Data]",
        "Cluster Split (80/20)\n[Residual OOF (20%)]",
        "DataSAIL Split (80/20)\n[Residual OOF (20%)]",
    ]

    BASELINE_MODEL: str = "ESM-2 (8M) - KNN"
    BEST_MODEL: str = "Fused (ProtT5 + ESM2) - Ridge"

    # Color definitions
    CMAP_BASE_COLOR: str = "#C4677E"  # Dusty pink heatmap base
    PISTACHIO_BASELINE_COLOR: str = "#70a845"  # Muted Pistachio Green
    PISTACHIO_BEST_COLOR: str = "#80b918"  # Vibrant Pistachio Green


def format_model_label(model_id: str) -> str:
    """Insert line break between representation and model name for compact y-axis labels."""
    return model_id.replace(" - ", "\n- ")


def load_and_prepare_data(config: Config) -> pd.DataFrame:
    """Load cached ablation results and normalize column identifiers."""
    if not config.RESULTS_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Results cache file missing at: {config.RESULTS_CACHE_PATH}"
        )

    df = pd.read_csv(config.RESULTS_CACHE_PATH)
    df["Pearson r"] = pd.to_numeric(df["Pearson r"], errors="coerce")

    df["Model_ID"] = df["Representation"] + " - " + df["Model"]
    df["Scenario_Label"] = (
        df["Evaluation Protocol"].str.replace(" (Full Dataset)", "", regex=False)
        + "\n["
        + df["Cleaning Method"]
        + "]"
    )

    df_subset = df[
        df["Model_ID"].isin(config.SELECTED_MODELS)
        & df["Scenario_Label"].isin(config.SELECTED_SCENARIOS)
    ].copy()

    return df_subset


def plot_summary_heatmap(df_subset: pd.DataFrame, config: Config) -> None:
    """Render publication-ready 5x5 summary ablation heatmap with larger, high-density typography."""
    config.OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    sns.set_theme(style="white", font="sans-serif")

    pivot_df = df_subset.pivot_table(
        index="Model_ID",
        columns="Scenario_Label",
        values="Pearson r",
        aggfunc="mean",
    )

    ordered_rows = [m for m in config.SELECTED_MODELS if m in pivot_df.index]
    ordered_cols = [s for s in config.SELECTED_SCENARIOS if s in pivot_df.columns]
    pivot_df = pivot_df.reindex(index=ordered_rows, columns=ordered_cols)

    n_cols = len(ordered_cols)

    # Compact square canvas size
    fig, ax = plt.subplots(figsize=(8.5, 8.2), dpi=300)
    cmap = sns.light_palette(config.CMAP_BASE_COLOR, as_cmap=True)

    sns.heatmap(
        pivot_df,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        linewidths=1.8,
        linecolor="#ffffff",
        square=True,
        annot_kws={"size": 13.5, "weight": "bold"},  # Powiększone, czytelne cyfry
        cbar_kws={
            "label": "Pearson correlation (r)",
            "shrink": 0.72,
            "pad": 0.035,
        },
    )

    # Formatowanie paska skali
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(
        "Pearson correlation (r)", fontsize=11, fontweight="bold", labelpad=8
    )

    ax.set_xlabel(
        "Evaluation Protocol & Data Cleaning Strategy",
        fontsize=12,
        fontweight="bold",
        labelpad=12,
    )
    ax.set_ylabel(
        "Model & Feature Representation",
        fontsize=12,
        fontweight="bold",
        labelpad=12,
    )

    # Powiększone i pogrubione etykiety osi X (zwięzły kąt 28 stopnia)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=28,
        ha="right",
        rotation_mode="anchor",
        fontsize=10.5,
        fontweight="bold",
    )

    # Powiększone i ściśnięte wieloliniowo etykiety osi Y
    formatted_ytick_labels = []
    for label in ax.get_yticklabels():
        raw_text = label.get_text()
        formatted_text = format_model_label(raw_text)
        if config.BASELINE_MODEL and raw_text == config.BASELINE_MODEL:
            formatted_text += "\n(baseline)"
        elif config.BEST_MODEL and raw_text == config.BEST_MODEL:
            formatted_text += "\n(best)"
        formatted_ytick_labels.append(formatted_text)

    ax.set_yticklabels(
        formatted_ytick_labels, rotation=0, fontsize=10.5, fontweight="bold"
    )

    # Ramki obrysowujące dla baseline i championa
    highlights = [
        (config.BASELINE_MODEL, config.PISTACHIO_BASELINE_COLOR),
        (config.BEST_MODEL, config.PISTACHIO_BEST_COLOR),
    ]

    for row_label, pistachio_color in highlights:
        if row_label and row_label in ordered_rows:
            row_idx = ordered_rows.index(row_label)
            ax.add_patch(
                plt.Rectangle(
                    (0, row_idx),
                    n_cols,
                    1,
                    fill=False,
                    edgecolor=pistachio_color,
                    lw=3.0,
                    clip_on=False,
                )
            )

    ax.set_title(
        "ACEpIC Ablation Summary: Baseline vs. Best Across Protocols",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )

    plt.tight_layout()
    fig.savefig(config.OUTPUT_PLOT_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(config.OUTPUT_PDF_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Summary PNG saved successfully to: {config.OUTPUT_PLOT_PATH}")
    print(f"Summary PDF saved successfully to: {config.OUTPUT_PDF_PATH}")


def main() -> None:
    """Execution entry point."""
    try:
        config = Config()
        print("Loading and preparing ablation summary data...")
        df_subset = load_and_prepare_data(config)

        print("Generating compressed summary heatmap with larger typography...")
        plot_summary_heatmap(df_subset, config)

    except Exception as error:
        print(f"Execution failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()