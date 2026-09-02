import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr

# Try importing logomaker for Sequence Logos; fallback to Seaborn heatmaps if not available
try:
    import logomaker
    HAS_LOGOMAKER = True
except ImportError:
    HAS_LOGOMAKER = False

# Paths configuration
base_dir = "/home/marta/Pulpit/ACEpIC"
csv_path = os.path.join(base_dir, "embeddings_data/cleaned_peptides.csv")
output_dir = os.path.join(base_dir, "scripts/visualization/plots")
os.makedirs(output_dir, exist_ok=True)

# Define embedding models
models = [
    {"name": "ESM-2 (8M)", "file": "esm2_8m_embeddings.npy"},
    {"name": "ESM-2 (35M)", "file": "esm2_35m_embeddings.npy"},
    {"name": "ESM-2 (650M)", "file": "esm2_650m_embeddings.npy"},
    {"name": "ProtT5-XL", "file": "prott5_embeddings.npy"},
    {"name": "Ankh Base", "file": "ankh_base_embeddings.npy"},
]

# Load dataset and clean NaNs
df = pd.read_csv(csv_path)
valid_mask = df["pIC50"].notna()
df_clean = df[valid_mask].copy().reset_index(drop=True)
y_valid = df_clean["pIC50"].values


# ==========================================
# 1. PCA & EXPLAINED VARIANCE ANALYSIS
# ==========================================
print("=== Starting 1. PCA & Explained Variance Analysis ===")

fig_pca, axes_pca = plt.subplots(len(models), 2, figsize=(14, 4 * len(models)))

for idx, model_info in enumerate(models):
    emb_path = os.path.join(base_dir, "embeddings_data", model_info["file"])
    if not os.path.exists(emb_path):
        print(f"[SKIP] Embedding file missing: {model_info['file']}")
        continue

    X = np.load(emb_path)[valid_mask.values]
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute 50 components PCA
    n_components = min(50, X_clean.shape[0], X_clean.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_clean)

    cum_variance = np.cumsum(pca.explained_variance_ratio_) * 100

    # Subplot A: Cumulative Variance
    ax_var = axes_pca[idx, 0]
    ax_var.plot(range(1, n_components + 1), cum_variance, marker="o", linestyle="--", color="navy", ms=3)
    ax_var.axhline(y=80, color="r", linestyle=":", label="80% Variance threshold")
    ax_var.set_title(f"{model_info['name']}: Cumulative Explained Variance", fontsize=11, fontweight="bold")
    ax_var.set_xlabel("Number of Principal Components")
    ax_var.set_ylabel("Explained Variance (%)")
    ax_var.legend(loc="lower right")
    ax_var.grid(True, alpha=0.3)

    # Subplot B: PC1 vs PC2 Scatter Plot
    ax_scatter = axes_pca[idx, 1]
    sc = ax_scatter.scatter(X_pca[:, 0], X_pca[:, 1], c=y_valid, cmap="viridis", alpha=0.7, s=15)
    plt.colorbar(sc, ax=ax_scatter, label="pIC50")
    ax_scatter.set_title(f"{model_info['name']}: PC1 vs PC2 Projection", fontsize=11, fontweight="bold")
    ax_scatter.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax_scatter.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

plt.tight_layout()
pca_plot_path = os.path.join(output_dir, "pca_variance_and_projections.png")
fig_pca.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
plt.close(fig_pca)
print(f"[OK] Saved PCA analysis plot: {pca_plot_path}\n")


# ==========================================
# 2. COSINE SIMILARITY VS |DELTA pIC50|
# ==========================================
print("=== Starting 2. Cosine Similarity vs |Delta pIC50| Analysis ===")

def compute_pairwise_cosine_and_delta(X, y, num_samples=50000, seed=42):
    """Vectorized sampling of pairwise cosine similarities and target differences."""
    np.random.seed(seed)
    n = len(y)
    idx1 = np.random.randint(0, n, size=num_samples)
    idx2 = np.random.randint(0, n, size=num_samples)
    
    # Filter out self-pairs
    valid_pairs = idx1 != idx2
    idx1, idx2 = idx1[valid_pairs], idx2[valid_pairs]

    # L2 normalize rows for fast cosine calculation
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    # Vectorized Dot Product
    cosine_sims = np.sum(X_norm[idx1] * X_norm[idx2], axis=1)
    delta_y = np.abs(y[idx1] - y[idx2])

    return cosine_sims, delta_y

fig_cos, axes_cos = plt.subplots(2, 3, figsize=(18, 11))
axes_cos = axes_cos.flatten()

for idx, model_info in enumerate(models):
    emb_path = os.path.join(base_dir, "embeddings_data", model_info["file"])
    if not os.path.exists(emb_path):
        continue

    X = np.load(emb_path)[valid_mask.values]
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cos_sims, delta_y = compute_pairwise_cosine_and_delta(X_clean, y_valid)

    # Compute correlations
    spearman_rho, _ = spearmanr(cos_sims, delta_y)
    pearson_r, _ = pearsonr(cos_sims, delta_y)

    ax = axes_cos[idx]
    # Hexbin plot for high density pairwise data
    hb = ax.hexbin(cos_sims, delta_y, gridsize=40, cmap="Blues", mincnt=1)
    cb = fig_cos.colorbar(hb, ax=ax, label="Count")

    # Add trend line
    sns.regplot(
        x=cos_sims, y=delta_y, ax=ax, scatter=False, 
        color="crimson", line_kws={"linewidth": 2, "linestyle": "--"}
    )

    ax.set_title(
        f"{model_info['name']}\nSpearman ρ: {spearman_rho:.3f} | Pearson r: {pearson_r:.3f}",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("|Δ pIC50|")

fig_cos.delaxes(axes_cos[5])
plt.tight_layout()
cos_plot_path = os.path.join(output_dir, "cosine_sim_vs_delta_pic50.png")
fig_cos.savefig(cos_plot_path, dpi=300, bbox_inches="tight")
plt.close(fig_cos)
print(f"[OK] Saved Cosine Similarity validation plot: {cos_plot_path}\n")


# ==========================================
# 3. BIOLOGICAL MOTIF ANALYSIS (TOP 10% vs BOTTOM 10%)
# ==========================================
print("=== Starting 3. Biological Motif Analysis ===")

# Extract Top and Bottom 10% peptides
top_10_threshold = df_clean["pIC50"].quantile(0.90)
bottom_10_threshold = df_clean["pIC50"].quantile(0.10)

top_peptides = df_clean[df_clean["pIC50"] >= top_10_threshold]["sequence"].tolist()
bottom_peptides = df_clean[df_clean["pIC50"] <= bottom_10_threshold]["sequence"].tolist()

print(f"Top 10% Active Peptides Count: {len(top_peptides)} (pIC50 >= {top_10_threshold:.2f})")
print(f"Bottom 10% Least Active Count: {len(bottom_peptides)} (pIC50 <= {bottom_10_threshold:.2f})")

# ACE Inhibitor Activity is heavily driven by C-terminal residues (C-1, C-2, C-3, C-4)
def build_terminal_matrix(sequences, length=4, terminal="C"):
    """Creates position-frequency dataframe for C-terminal or N-terminal amino acids."""
    standard_aa = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
    counts = {aa: [0] * length for aa in standard_aa}

    for seq in sequences:
        if len(seq) < length:
            continue
        subseq = seq[-length:] if terminal == "C" else seq[:length]
        for pos, aa in enumerate(subseq):
            if aa in counts:
                counts[aa][pos] += 1

    df_freq = pd.DataFrame(counts, index=[f"C-{length-i-1}" if terminal == "C" else f"N+{i+1}" for i in range(length)])
    # Convert counts to probabilities/percentages
    df_prob = df_freq.div(df_freq.sum(axis=1), axis=0).fillna(0)
    return df_prob

top_c_term = build_terminal_matrix(top_peptides, length=4, terminal="C")
bottom_c_term = build_terminal_matrix(bottom_peptides, length=4, terminal="C")

# Generate Sequence Logos or Heatmap Fallback
if HAS_LOGOMAKER:
    fig_logo, axes_logo = plt.subplots(2, 1, figsize=(10, 6))

    logomaker.Logo(top_c_term, ax=axes_logo[0], color_scheme="chemistry")
    axes_logo[0].set_title("C-Terminal Sequence Logo: Top 10% Active Peptides", fontsize=12, fontweight="bold")
    axes_logo[0].set_ylabel("Probability")

    logomaker.Logo(bottom_c_term, ax=axes_logo[1], color_scheme="chemistry")
    axes_logo[1].set_title("C-Terminal Sequence Logo: Bottom 10% Inactive Peptides", fontsize=12, fontweight="bold")
    axes_logo[1].set_ylabel("Probability")

    plt.tight_layout()
    logo_path = os.path.join(output_dir, "sequence_logo_top_vs_bottom.png")
    fig_logo.savefig(logo_path, dpi=300, bbox_inches="tight")
    plt.close(fig_logo)
    print(f"[OK] Saved Sequence Logo plot: {logo_path}")

# Always save Position-Frequency Heatmap Comparison
fig_hm, axes_hm = plt.subplots(1, 2, figsize=(16, 5))

sns.heatmap(top_c_term.T, ax=axes_hm[0], cmap="YlGnBu", annot=True, fmt=".2f", cbar=True)
axes_hm[0].set_title("C-Terminal AA Frequencies: Top 10% Active", fontsize=12, fontweight="bold")
axes_hm[0].set_xlabel("Position relative to C-terminus")
axes_hm[0].set_ylabel("Amino Acid")

sns.heatmap(bottom_c_term.T, ax=axes_hm[1], cmap="OrRd", annot=True, fmt=".2f", cbar=True)
axes_hm[1].set_title("C-Terminal AA Frequencies: Bottom 10% Inactive", fontsize=12, fontweight="bold")
axes_hm[1].set_xlabel("Position relative to C-terminus")
axes_hm[1].set_ylabel("Amino Acid")

plt.tight_layout()
heatmap_path = os.path.join(output_dir, "cterminal_frequency_heatmap_top_vs_bottom.png")
fig_hm.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close(fig_hm)
print(f"[OK] Saved C-Terminal Frequency Heatmap plot: {heatmap_path}")

print("\n=== All Advanced EDA Tasks Completed Successfully! ===")