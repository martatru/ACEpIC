import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directory paths
base_dir = "/home/marta/Pulpit/ACEpIC"
csv_path = os.path.join(base_dir, "embeddings_data/cleaned_peptides.csv")
output_dir = os.path.join(base_dir, "scripts/visualization/plots")

os.makedirs(output_dir, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(csv_path)

# Filter out missing IC50 / sequence rows
ic50_col = "IC50 (µM)" if "IC50 (µM)" in df.columns else None
if not ic50_col:
    raise KeyError("Column 'IC50 (µM)' not found in dataset.")

df[ic50_col] = pd.to_numeric(df[ic50_col], errors="coerce")
df_valid = df.dropna(subset=["sequence", ic50_col]).copy()
df_valid["seq_len"] = df_valid["sequence"].str.len()

# Filter out non-positive IC50 for clean statistics
df_valid = df_valid[df_valid[ic50_col] > 0]

# 2. Setup Plot Grid (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Peptide Length Count Distribution (Bar Plot)
len_counts = df_valid["seq_len"].value_counts().sort_index()
sns.barplot(
    x=len_counts.index, 
    y=len_counts.values, 
    ax=axes[0], 
    palette="viridis",
    hue=len_counts.index,
    legend=False
)
axes[0].set_title("Peptide Count by Sequence Length", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Sequence Length (Number of Amino Acids)", fontsize=10)
axes[0].set_ylabel("Number of Peptides", fontsize=10)
axes[0].grid(axis="y", linestyle=":", alpha=0.6)

# Annotate exact counts on top of bars
for p in axes[0].patches:
    height = int(p.get_height())
    if height > 0:
        axes[0].annotate(
            f"{height}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points"
        )

# Plot 2: Raw IC50 Distribution (Linear Scale)
sns.histplot(
    df_valid[ic50_col], 
    bins=40, 
    kde=True, 
    ax=axes[1], 
    color="darkcrimson" if hasattr(sns, "darkcrimson") else "crimson"
)
axes[1].set_title("Raw IC50 Distribution (Linear Scale)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("IC50 (µM)", fontsize=10)
axes[1].set_ylabel("Number of Peptides", fontsize=10)
axes[1].grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
plot_path = os.path.join(output_dir, "raw_ic50_and_length_counts.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] Saved plot to: {plot_path}")