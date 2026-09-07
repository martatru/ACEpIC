import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CLEANED_DATASET_PATH = "embeddings_data/cleaned_peptides.csv"
DATASET_DIR = "dataset"
OUTPUT_PLOT_PATH = "scripts/visualization/plots/cleanlab_removed_eda.png"


def find_raw_dataset_path(dataset_dir: str) -> str:
    """Locate the raw CSV dataset file inside the dataset directory."""
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directory not found: {dataset_dir}")

    csv_files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    return csv_files[0]


def extract_removed_peptides(
    raw_path: str, cleaned_path: str
) -> pd.DataFrame:
    """Identify peptides present in raw dataset but missing in cleaned dataset."""
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_path}")

    df_raw = pd.read_csv(raw_path)
    df_clean = pd.read_csv(cleaned_path)

    seq_col = "sequence" if "sequence" in df_raw.columns else df_raw.columns[0]

    removed_mask = ~df_raw[seq_col].isin(df_clean[seq_col])
    df_removed = df_raw[removed_mask].copy()

    if "length" not in df_removed.columns:
        df_removed["length"] = df_removed[seq_col].astype(str).str.len()

    if "pIC50" not in df_removed.columns and "IC50" in df_removed.columns:
        df_removed["pIC50"] = 6 - np.log10(df_removed["IC50"])

    if "IC50" not in df_removed.columns and "pIC50" in df_removed.columns:
        df_removed["IC50"] = 10 ** (6 - df_removed["pIC50"])

    print(
        f"Raw dataset total: {len(df_raw)} | Cleaned dataset total: {len(df_clean)}"
    )
    print(f"Extracted {len(df_removed)} removed samples for EDA analysis.")

    return df_removed


def create_cleanlab_eda_matrix(
    df_removed: pd.DataFrame, output_path: str
) -> None:
    """Generate 2x2 EDA plots for removed dataset samples."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="white", font="sans-serif")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Panel 1: Sequence Length Distribution
    ax1 = axes[0, 0]
    ax1.grid(False)
    length_counts = df_removed["length"].value_counts().sort_index()

    sns.barplot(
        x=length_counts.index,
        y=length_counts.values,
        ax=ax1,
        palette="mako",
        edgecolor="#334155",
        linewidth=0.6,
    )
    ax1.set_xlabel("Sequence Length (aa)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Peptide Count", fontsize=11, labelpad=8)
    ax1.set_title(
        "Removed Samples: Count by Sequence Length",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    for p in ax1.patches:
        val = int(p.get_height())
        if val > 0:
            ax1.annotate(
                f"{val}",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    # Panel 2: Raw IC50 Distribution
    ax2 = axes[0, 1]
    ax2.grid(False)
    if "IC50" in df_removed.columns:
        sns.histplot(
            df_removed["IC50"],
            ax=ax2,
            kde=True,
            color="#e11d48",
            bins=30,
            stat="count",
        )
        ax2.set_xlabel("IC50 (µM)", fontsize=11, labelpad=8)
        ax2.set_ylabel("Peptide Count", fontsize=11, labelpad=8)
        ax2.set_title(
            "Removed Samples: Raw IC50 Distribution",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

    # Panel 3: pIC50 Target Distribution
    ax3 = axes[1, 0]
    ax3.grid(False)
    if "pIC50" in df_removed.columns:
        sns.histplot(
            df_removed["pIC50"],
            ax=ax3,
            kde=True,
            color="#0d9488",
            bins=25,
            stat="count",
        )
        ax3.set_xlabel("pIC50 Target Value", fontsize=11, labelpad=8)
        ax3.set_ylabel("Peptide Count", fontsize=11, labelpad=8)
        ax3.set_title(
            "Removed Samples: pIC50 Distribution",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

    # Panel 4: pIC50 vs Sequence Length Boxplot
    ax4 = axes[1, 1]
    ax4.grid(False)
    if "pIC50" in df_removed.columns:
        sns.boxplot(
            data=df_removed,
            x="length",
            y="pIC50",
            ax=ax4,
            palette="crest",
            fliersize=3,
            linewidth=0.8,
        )
        ax4.set_xlabel("Sequence Length (aa)", fontsize=11, labelpad=8)
        ax4.set_ylabel("pIC50 Target Value", fontsize=11, labelpad=8)
        ax4.set_title(
            "Removed Samples: pIC50 vs Sequence Length",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Cleanlab removed samples EDA saved successfully to: {output_path}")


def main() -> None:
    """Execution pipeline for plotting removed dataset samples."""
    try:
        raw_path = find_raw_dataset_path(DATASET_DIR)
        print(f"Using raw dataset file: {raw_path}")

        df_removed = extract_removed_peptides(raw_path, CLEANED_DATASET_PATH)
        create_cleanlab_eda_matrix(df_removed, OUTPUT_PLOT_PATH)

    except Exception as error:
        print(f"Execution failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()