import os
import sys
import warnings
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm
from xgboost import XGBRegressor

# Global warning suppression for optimization edge-cases
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Config:
    """Central configuration for paths and execution settings."""
    DATASET_PATH: str = "embeddings_data/cleaned_peptides.csv"
    DATASAYL_SPLIT_PATH: str = "embeddings_data/datasail_split.csv"
    EMBEDDINGS_DIR: str = "embeddings_data"
    RESULTS_CACHE_PATH: str = "embeddings_data/master_ablation_results.csv"
    OUTPUT_HEATMAP_PATH: str = "scripts/visualization/plots/master_ablation_heatmap.png"

    EMBEDDING_FILES: Dict[str, str] = {
        "ProtT5-XL": "prott5_embeddings.npy",
        "ESM-2 (650M)": "esm2_650m_embeddings.npy",
        "ESM-2 (35M)": "esm2_35m_embeddings.npy",
        "ESM-2 (8M)": "esm2_8m_embeddings.npy",
        "Ankh Base": "ankh_base_embeddings.npy",
    }
    RANDOM_SEED: int = 42


class DataLoader:
    """Loads dataset targets and pre-computed protein language embeddings."""

    def __init__(self, config: Config = Config()):
        self.config = config

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
        if not os.path.exists(self.config.DATASET_PATH):
            raise FileNotFoundError(f"Dataset CSV missing: {self.config.DATASET_PATH}")

        df = pd.read_csv(self.config.DATASET_PATH)
        if "pIC50" not in df.columns:
            raise KeyError("Target column 'pIC50' missing from dataset CSV.")

        y = df["pIC50"].values
        embeddings = {}

        for label, filename in self.config.EMBEDDING_FILES.items():
            filepath = os.path.join(self.config.EMBEDDINGS_DIR, filename)
            if os.path.exists(filepath):
                embeddings[label] = np.load(filepath)

        if "ProtT5-XL" in embeddings and "ESM-2 (650M)" in embeddings:
            fused = np.hstack((embeddings["ProtT5-XL"], embeddings["ESM-2 (650M)"]))
            embeddings["Fused (ProtT5 + ESM2)"] = fused

        return df, y, embeddings


class DatasetSplitter:
    """Handles train/test split generation."""

    def __init__(self, seed: int = Config.RANDOM_SEED):
        self.seed = seed

    def cluster_split(
        self, X_ref: np.ndarray, test_size: float = 0.20, threshold: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        distances = pdist(X_ref, metric="cosine")
        clusters = fcluster(
            linkage(distances, method="complete"), t=threshold, criterion="distance"
        )

        unique_clusters = np.unique(clusters)
        np.random.seed(self.seed)
        np.random.shuffle(unique_clusters)

        n_test = int(len(X_ref) * test_size)
        test_idx, train_idx = [], []
        current_test_count = 0

        for cluster in unique_clusters:
            indices = np.where(clusters == cluster)[0]
            if current_test_count < n_test:
                test_idx.extend(indices)
                current_test_count += len(indices)
            else:
                train_idx.extend(indices)

        return np.array(train_idx), np.array(test_idx)

    def get_indices(
        self, df: pd.DataFrame, X_ref: np.ndarray, split_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.arange(len(df))

        if split_type == "Random Split (80/20)":
            return train_test_split(indices, test_size=0.20, random_state=self.seed)

        if split_type == "Cluster Split (80/20)":
            return self.cluster_split(X_ref, test_size=0.20)

        if split_type == "DataSAIL Split (80/20)":
            if os.path.exists(Config.DATASAYL_SPLIT_PATH):
                df_split = pd.read_csv(Config.DATASAYL_SPLIT_PATH)
                if "split" in df_split.columns:
                    train_mask = (df_split["split"] == "train").values
                    test_mask = (df_split["split"] == "test").values
                    return np.where(train_mask)[0], np.where(test_mask)[0]

            return self.cluster_split(X_ref, test_size=0.20)

        raise ValueError(f"Unknown split strategy: {split_type}")


class DatasetCleaner:
    """Applies continuous dataset cleaning protocols."""

    @staticmethod
    def get_clean_mask(X: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
        n_samples = len(y)

        if method == "Raw Data":
            return np.ones(n_samples, dtype=bool)

        if "Residual OOF" in method:
            cutoff = float(method.split("(")[1].split("%")[0])
            kf = KFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)
            residuals = np.zeros_like(y, dtype=np.float64)

            for train_fold, val_fold in kf.split(X, y):
                model = Ridge(alpha=1.0)
                model.fit(X[train_fold], y[train_fold])
                preds = model.predict(X[val_fold])
                residuals[val_fold] = np.abs(y[val_fold] - preds)

            threshold = np.percentile(residuals, 100.0 - cutoff)
            return residuals <= threshold

        if method == "Huber Robust (15%)":
            # Feature scaling + robust convergence parameters (max_iter=5000, tol=1e-3)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            huber = HuberRegressor(max_iter=5000, tol=1e-3)
            huber.fit(X_scaled, y)
            residuals = np.abs(y - huber.predict(X_scaled))
            threshold = np.percentile(residuals, 85.0)
            return residuals <= threshold

        if method == "Activity Cliffs":
            sim_matrix = cosine_similarity(X)
            noisy_indices = set()
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if sim_matrix[i, j] >= 0.95 and abs(y[i] - y[j]) >= 2.0:
                        noisy_indices.add(i)
                        noisy_indices.add(j)
            mask = np.ones(n_samples, dtype=bool)
            mask[list(noisy_indices)] = False
            return mask

        if method == "Ensemble Variance (15%)":
            model = ExtraTreesRegressor(
                n_estimators=100, random_state=Config.RANDOM_SEED, n_jobs=-1
            )
            model.fit(X, y)
            tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
            variances = np.var(tree_preds, axis=0)
            threshold = np.percentile(variances, 85.0)
            return variances <= threshold

        return np.ones(n_samples, dtype=bool)


class AblationBenchmark:
    """Executes fresh master ablation benchmark with progress tracking."""

    def __init__(self, config: Config = Config()):
        self.config = config
        self.splitter = DatasetSplitter()
        self.cleaner = DatasetCleaner()

    @staticmethod
    def get_models() -> Dict[str, object]:
        return {
            "Ridge": Ridge(alpha=1.0),
            "XGBoost": XGBRegressor(
                n_estimators=100, learning_rate=0.05, random_state=Config.RANDOM_SEED, n_jobs=-1
            ),
            "SVR": SVR(C=1.0, kernel="rbf"),
            "MLP": MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=1000,
                early_stopping=True,
                random_state=Config.RANDOM_SEED,
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=Config.RANDOM_SEED, n_jobs=-1
            ),
            "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        }

    def run(
        self, df: pd.DataFrame, y: np.ndarray, embeddings: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        protocols = [
            "10-Fold CV (Full Dataset)",
            "Random Split (80/20)",
            "Cluster Split (80/20)",
            "DataSAIL Split (80/20)",
        ]

        cleaning_methods = [
            "Raw Data",
            "Residual OOF (10%)",
            "Residual OOF (15%)",
            "Residual OOF (20%)",
            "Huber Robust (15%)",
            "Activity Cliffs",
            "Ensemble Variance (15%)",
        ]

        models = self.get_models()
        X_ref = embeddings.get("ProtT5-XL", list(embeddings.values())[0])
        records = []

        total_tasks = (
            len(protocols) * len(embeddings) * len(cleaning_methods) * len(models)
        )

        print(f"Starting complete master ablation recalculation ({total_tasks} evaluations)...")

        with tqdm(total=total_tasks, desc="Ablation Benchmark", unit="eval") as pbar:
            for protocol in protocols:
                if protocol == "10-Fold CV (Full Dataset)":
                    kf = KFold(n_splits=10, shuffle=True, random_state=Config.RANDOM_SEED)
                    for emb_name, X_full in embeddings.items():
                        for method in cleaning_methods:
                            mask = self.cleaner.get_clean_mask(X_full, y, method)
                            X_sub, y_sub = X_full[mask], y[mask]

                            for model_name, model_obj in models.items():
                                pearsons = []
                                for train_idx, val_idx in kf.split(X_sub, y_sub):
                                    model_obj.fit(X_sub[train_idx], y_sub[train_idx])
                                    preds = model_obj.predict(X_sub[val_idx])
                                    r_val, _ = pearsonr(y_sub[val_idx], preds)
                                    pearsons.append(r_val)

                                records.append(
                                    {
                                        "Evaluation Protocol": protocol,
                                        "Representation": emb_name,
                                        "Model": model_name,
                                        "Cleaning Method": method,
                                        "Pearson r": float(np.mean(pearsons)),
                                    }
                                )
                                pbar.update(1)

                else:
                    train_idx, test_idx = self.splitter.get_indices(df, X_ref, protocol)

                    for emb_name, X_full in embeddings.items():
                        X_train_full, X_test = X_full[train_idx], X_full[test_idx]
                        y_train_full, y_test = y[train_idx], y[test_idx]

                        for method in cleaning_methods:
                            clean_mask = self.cleaner.get_clean_mask(
                                X_train_full, y_train_full, method
                            )
                            X_train_clean = X_train_full[clean_mask]
                            y_train_clean = y_train_full[clean_mask]

                            for model_name, model_obj in models.items():
                                model_obj.fit(X_train_clean, y_train_clean)
                                preds = model_obj.predict(X_test)
                                r_val, _ = pearsonr(y_test, preds)

                                records.append(
                                    {
                                        "Evaluation Protocol": protocol,
                                        "Representation": emb_name,
                                        "Model": model_name,
                                        "Cleaning Method": method,
                                        "Pearson r": float(r_val),
                                    }
                                )
                                pbar.update(1)

        df_results = pd.DataFrame(records)
        os.makedirs(os.path.dirname(self.config.RESULTS_CACHE_PATH), exist_ok=True)
        df_results.to_csv(self.config.RESULTS_CACHE_PATH, index=False)
        print(f"\nFresh benchmark results saved to: {self.config.RESULTS_CACHE_PATH}")
        return df_results


class AblationVisualizer:
    """Generates master heatmap visualization."""

    @staticmethod
    def plot_heatmap(df_results: pd.DataFrame, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sns.set_theme(style="white", font="sans-serif")

        df_results["Pearson r"] = pd.to_numeric(df_results["Pearson r"], errors="coerce")

        pivot_df = df_results.pivot_table(
            index=["Representation", "Model"],
            columns=["Evaluation Protocol", "Cleaning Method"],
            values="Pearson r",
            aggfunc="mean",
        ).astype(float)

        fig, ax = plt.subplots(figsize=(24, 11), dpi=300)
        cmap = sns.light_palette("#6d28d9", as_cmap=True)

        sns.heatmap(
            pivot_df,
            ax=ax,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            linewidths=0.5,
            linecolor="#ffffff",
            annot_kws={"size": 6.5, "weight": "bold"},
            cbar_kws={"label": "Pearson Correlation (r)", "shrink": 0.8},
        )

        ax.set_title(
            "Master Ablation Study: Performance Across Representations, Protocols & Data Cleaning Strategies",
            fontsize=14,
            fontweight="bold",
            pad=16,
        )
        ax.set_xlabel("Evaluation Protocol & Data Cleaning Strategy", fontsize=11, labelpad=10)
        ax.set_ylabel("Representation & Regressor", fontsize=11, labelpad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Master heatmap exported to: {output_path}")


def main() -> None:
    try:
        config = Config()
        loader = DataLoader(config)
        df, y, embeddings = loader.load_data()

        benchmark = AblationBenchmark(config)
        df_results = benchmark.run(df, y, embeddings)

        visualizer = AblationVisualizer()
        visualizer.plot_heatmap(df_results, config.OUTPUT_HEATMAP_PATH)

    except Exception as error:
        print(f"Pipeline execution failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()