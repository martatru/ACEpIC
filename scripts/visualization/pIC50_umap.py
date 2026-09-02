import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP

# Directory paths
base_dir = "/home/marta/Pulpit/ACEpIC"
csv_path = os.path.join(base_dir, "embeddings_data/cleaned_peptides.csv")
cleanlab_path = os.path.join(base_dir, "embeddings_data/cleanlab_issues_report.csv")
output_dir = os.path.join(base_dir, "scripts/visualization/plots")

os.makedirs(output_dir, exist_ok=True)

# Toggle to filter out Cleanlab issues (False = Raw data, True = Cleaned data)
USE_CLEANLAB = False

# Models configuration
models = [
    {"name": "ESM-2 (8M)", "file": "esm2_8m_embeddings.npy"},
    {"name": "ESM-2 (35M)", "file": "esm2_35m_embeddings.npy"},
    {"name": "ESM-2 (650M)", "file": "esm2_650m_embeddings.npy"},
    {"name": "ProtT5-XL", "file": "prott5_embeddings.npy"},
    {"name": "Ankh Base", "file": "ankh_base_embeddings.npy"},
]

# 1. Load dataset and filter valid pIC50 entries
df = pd.read_csv(csv_path)
valid_mask = df["pIC50"].notna()

# Optional: Apply Cleanlab noise filtering
if USE_CLEANLAB and os.path.exists(cleanlab_path):
    cleanlab_df = pd.read_csv(cleanlab_path)
    if "is_issue" in cleanlab_df.columns:
        valid_mask = valid_mask & (~cleanlab_df["is_issue"].astype(bool))
        print("[INFO] Cleanlab issue filtering APPLIED.")

df_clean = df[valid_mask].copy()
y_valid = df_clean["pIC50"].values

# Standardize color scale range across all subplots
vmin, vmax = np.nanmin(y_valid), np.nanmax(y_valid)

# 2. Grid plot setup (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

print(f"Processing UMAP projections for {len(models)} embedding models...")

last_scatter = None

for idx, model_info in enumerate(models):
    emb_file_path = os.path.join(base_dir, "embeddings_data", model_info["file"])
    ax = axes[idx]

    if not os.path.exists(emb_file_path):
        print(f"[SKIP] File not found: {model_info['file']}")
        ax.set_title(f"{model_info['name']} (Missing)", fontsize=12, fontweight="bold")
        ax.axis("off")
        continue

    print(f"-> Computing UMAP for: {model_info['name']}...")
    X = np.load(emb_file_path)
    X_valid = X[valid_mask.values]

    # Clean NaNs/Infs if present in extraction
    X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)

    # Run UMAP
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(X_valid)

    # Plot on Grid Subplot
    last_scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=y_valid,
        cmap="viridis",
        alpha=0.75,
        s=14,
        vmin=vmin,
        vmax=vmax
    )
    ax.set_title(f"{model_info['name']}", fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)

# Remove unused 6th subplot
fig.delaxes(axes[5])

# Add global shared colorbar
if last_scatter is not None:
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_scatter, cax=cbar_ax)
    cbar.set_label("pIC50 Target Value", fontsize=11, fontweight="bold")

fig.suptitle("pLM Embedding Spaces Projection (UMAP)", fontsize=16, fontweight="bold", y=0.95)

suffix = "_cleanlab" if USE_CLEANLAB else ""
grid_path = os.path.join(output_dir, f"umap_all_models_comparison{suffix}.png")
plt.subplots_adjust(right=0.9, hspace=0.3, wspace=0.25)
plt.savefig(grid_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n[OK] Combined grid plot successfully saved to: {grid_path}")