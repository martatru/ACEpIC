from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Resolve repository root directory dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PLOT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "visualization"
    / "plots"
    / "predicted_ic50_by_variant.png"
)


def plot_predicted_ic50_by_variant() -> None:
    """Generate compact square bar chart of predicted IC50 values per peptide."""
    OUTPUT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Data extracted 1:1 from the reference plot
    data = {
        "Peptide": [
            "IDL",
            "IDF",
            "VAK",
            "ISL",
            "VSK",
            "IQL",
            "IVK",
            "ASF",
            "IVVL",
            "IVPL",
            "IAVL",
            "VPK",
            "VTK",
            "IPK",
            "APK",
        ],
        "IC50": [
            12.5,
            14.2,
            39.8,
            42.4,
            51.0,
            57.4,
            81.5,
            97.0,
            106.4,
            118.1,
            124.5,
            135.4,
            137.7,
            186.4,
            204.3,
        ],
    }
    df = pd.DataFrame(data)

    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    sns.set_theme(style="white", font="sans-serif")

    # Compact square canvas ratio (6.8 x 6.5 inches)
    fig, ax = plt.subplots(figsize=(6.8, 6.5), dpi=300)

    # Rosy pink bars with adjusted width
    bars = ax.bar(
        df["Peptide"],
        df["IC50"],
        color="#d2a0a6",
        width=0.68,
        edgecolor="none",
    )

    # Annotations on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 3.0,
            f"{yval:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#4a3b3a",
        )

    # Bold Title in Title Case
    ax.set_title(
        "Predicted IC50 for the Discovered Peptides",
        fontsize=12.5,
        fontweight="bold",
        color="#333333",
        pad=14,
    )

    # Bold Axis Labels in Title Case
    ax.set_xlabel(
        "Peptide Sequence",
        fontsize=12.5,
        fontweight="bold",
        color="#333333",
        labelpad=8,
    )
    ax.set_ylabel(
        "IC50 (µM)",
        fontsize=11,
        fontweight="bold",
        color="#333333",
        labelpad=8,
    )

    # Axis limits & aesthetics
    ax.set_ylim(0, 225)
    ax.tick_params(colors="#4a3b3a", labelsize=9)
    ax.set_xticklabels(
        df["Peptide"], rotation=30, ha="right", rotation_mode="anchor"
    )

    # Subtle spine styling
    ax.spines["left"].set_color("#8a9a8a")
    ax.spines["bottom"].set_color("#8a9a8a")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    sns.despine(top=True, right=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved successfully to: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    plot_predicted_ic50_by_variant()