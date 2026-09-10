import os
import sys
import warnings
from typing import Dict, Set, Tuple

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

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Config:
    """Central configuration for benchmark execution settings and paths."""

    DATASET_PATH: str = "embeddings_data/cleaned_peptides.csv"
    DATASAYL_SPLIT_PATH: str = "embeddings_data/datasail_split.csv"
    EMBEDDINGS_DIR: str = "embeddings_data"
    RESULTS_CACHE_PATH: str = "embeddings_data/master_ablation_results.csv"
    OUTPUT_HEATMAP_PATH: str = (
        "scripts/visualization/plots/master_ablation_heatmap.png"
    )

    EMBEDDING_FILES: Dict[str, str] = {
        "ProtT5-XL": "prott5_embeddings.npy",
        "ESM-2 (650M)": "esm2_650m_embeddings.npy",
        "ESM-2 (35M)": "esm2_35m_embeddings.npy",
        "ESM-2 (8M)": "esm2_8m_embeddings.npy",
        "Ankh Base": "ankh_base_embeddings.npy",
    }
    RANDOM_SEED: int = 42


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Pearson correlation safely without crashing on edge cases."""
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    try:
        r_val, _ = pearsonr(y_true, y_pred)
        return float(r_val) if not np.isnan(r_val) else 0.0
    except Exception:
        return 0.0


class DataLoader:
    """Handles dataset and protein language model embedding loading."""

    def __init__(self, config: Config = Config()):
        self.config = config

    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
        """Load peptide dataset targets and precomputed embeddings."""
        if not os.path.exists(self.config.DATASET_PATH):
            raise FileNotFoundError(
                f"Dataset CSV missing: {self.config.DATASET_PATH}"
            )

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
            fused = np.hstack(
                (embeddings["ProtT5-XL"], embeddings["ESM-2 (650M)"])
            )
            embeddings["Fused (ProtT5 + ESM2)"] = fused

        return df, y, embeddings


class DatasetSplitter:
    """Generates train and test indices for evaluation strategies."""

    def __init__(self, seed: int = Config.RANDOM_SEED):
        self.seed = seed

    def cluster_split(
        self, X_ref: np.ndarray, test_size: float = 0.20, threshold: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform hierarchical complete-linkage sequence cluster split."""
        distances = pdist(X_ref, metric="cosine")
        clusters = fcluster(
            linkage(distances, method="complete"),
            t=threshold,
            criterion="distance",
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

        train_arr, test_arr = np.array(train_idx), np.array(test_idx)
        if len(train_arr) < 5 or len(test_arr) < 5:
            return train_test_split(
                np.arange(len(X_ref)), test_size=test_size, random_state=self.seed
            )

        return train_arr, test_arr

    def get_indices(
        self, df: pd.DataFrame, X_ref: np.ndarray, split_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve dataset split indices."""
        indices = np.arange(len(df))

        if split_type == "Random Split (80/20)":
            return train_test_split(
                indices, test_size=0.20, random_state=self.seed
            )

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
    """Applies data cleaning strategies and calculates sample masks."""

    def __init__(self):
        self._mask_cache: Dict[str, np.ndarray] = {}

    def get_clean_mask(
        self, X: np.ndarray, y: np.ndarray, method: str, context_key: str = ""
    ) -> np.ndarray:
        """Compute or retrieve cached dataset cleaning boolean mask safely."""
        cache_key = f"{context_key}_{method}_{X.shape[0]}"
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        n_samples = len(y)
        if n_samples < 5 or method == "Raw Data":
            mask = np.ones(n_samples, dtype=bool)
            self._mask_cache[cache_key] = mask
            return mask

        if "Residual OOF" in method:
            cutoff = float(method.split("(")[1].split("%")[0])
            kf = KFold(
                n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED
            )
            residuals = np.zeros_like(y, dtype=np.float64)

            for train_fold, val_fold in kf.split(X, y):
                model = Ridge(alpha=1.0)
                model.fit(X[train_fold], y[train_fold])
                preds = model.predict(X[val_fold])
                residuals[val_fold] = np.abs(y[val_fold] - preds)

            threshold = np.percentile(residuals, 100.0 - cutoff)
            mask = residuals <= threshold

        elif method == "Huber Robust (15%)":
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            huber = HuberRegressor(max_iter=5000, tol=1e-3)
            huber.fit(X_scaled, y)
            residuals = np.abs(y - huber.predict(X_scaled))
            threshold = np.percentile(residuals, 85.0)
            mask = residuals <= threshold

        elif method == "Activity Cliffs":
            sim_matrix = cosine_similarity(X)
            y_diff = np.abs(y[:, None] - y[None, :])
            cliff_matrix = (sim_matrix >= 0.95) & (y_diff >= 2.0)
            np.fill_diagonal(cliff_matrix, False)
            noisy_indices = np.where(cliff_matrix.any(axis=1))[0]
            mask = np.ones(n_samples, dtype=bool)
            mask[noisy_indices] = False

        elif method == "Ensemble Variance (15%)":
            model = ExtraTreesRegressor(
                n_estimators=100,
                random_state=Config.RANDOM_SEED,
                n_jobs=-1,
            )
            model.fit(X, y)
            tree_preds = np.array(
                [tree.predict(X) for tree in model.estimators_]
            )
            variances = np.var(tree_preds, axis=0)
            threshold = np.percentile(variances, 85.0)
            mask = variances <= threshold

        else:
            mask = np.ones(n_samples, dtype=bool)

        if np.sum(mask) < 2:
            mask = np.ones(n_samples, dtype=bool)

        self._mask_cache[cache_key] = mask
        return mask


class AblationBenchmark:
    """Executes full ablation study benchmark with checkpointing and resume support."""

    def __init__(self, config: Config = Config()):
        self.config = config
        self.splitter = DatasetSplitter()
        self.cleaner = DatasetCleaner()

    @staticmethod
    def get_models() -> Dict[str, object]:
        """Instantiate regression benchmark models."""
        return {
            "Ridge": Ridge(alpha=1.0),
            "XGBoost": XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=Config.RANDOM_SEED,
                n_jobs=-1,
            ),
            "SVR": SVR(C=1.0, kernel="rbf"),
            "MLP": MLPRegressor(
                hidden_layer_sizes=(128, 64),
                max_iter=1000,
                early_stopping=True,
                random_state=Config.RANDOM_SEED,
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=Config.RANDOM_SEED,
                n_jobs=-1,
            ),
            "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        }

    def _load_existing_results(self) -> Tuple[pd.DataFrame, Set[Tuple[str, str, str, str]]]:
        """Load cached results to support resuming interrupted runs."""
        cache_path = self.config.RESULTS_CACHE_PATH
        if os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path)
                completed = set(
                    zip(
                        df["Evaluation Protocol"],
                        df["Representation"],
                        df["Model"],
                        df["Cleaning Method"],
                    )
                )
                print(f"Resuming run: loaded {len(completed)} completed evaluations from cache.")
                return df, completed
            except Exception as e:
                print(f"Warning: Could not load cache file ({e}). Starting fresh.")
        return pd.DataFrame(), set()

    def run(
        self, df: pd.DataFrame, y: np.ndarray, embeddings: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Execute full benchmark evaluation grid with checkpoint saving."""
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

        df_results, completed_tasks = self._load_existing_results()
        records = df_results.to_dict("records") if not df_results.empty else []

        total_tasks = (
            len(protocols)
            * len(embeddings)
            * len(cleaning_methods)
            * len(models)
        )

        print(f"Starting ablation benchmark ({total_tasks} total evaluations)...")

        with tqdm(
            total=total_tasks, desc="Ablation Benchmark", unit="eval"
        ) as pbar:
            if completed_tasks:
                pbar.update(len(completed_tasks))

            for protocol in protocols:
                if protocol == "10-Fold CV (Full Dataset)":
                    kf = KFold(
                        n_splits=10,
                        shuffle=True,
                        random_state=Config.RANDOM_SEED,
                    )
                    for emb_name, X_full in embeddings.items():
                        for method in cleaning_methods:
                            context_key = f"{protocol}_{emb_name}"
                            mask = self.cleaner.get_clean_mask(
                                X_full, y, method, context_key
                            )
                            X_sub, y_sub = X_full[mask], y[mask]

                            for model_name, model_obj in models.items():
                                task_key = (protocol, emb_name, model_name, method)
                                if task_key in completed_tasks:
                                    continue

                                pearsons = []
                                if len(y_sub) >= 2:
                                    for train_idx, val_idx in kf.split(
                                        X_sub, y_sub
                                    ):
                                        try:
                                            model_obj.fit(
                                                X_sub[train_idx],
                                                y_sub[train_idx],
                                            )
                                            preds = model_obj.predict(
                                                X_sub[val_idx]
                                            )
                                            pearsons.append(
                                                safe_pearsonr(
                                                    y_sub[val_idx], preds
                                                )
                                            )
                                        except Exception:
                                            pearsons.append(0.0)

                                score = (
                                    float(np.mean(pearsons))
                                    if pearsons
                                    else 0.0
                                )
                                records.append(
                                    {
                                        "Evaluation Protocol": protocol,
                                        "Representation": emb_name,
                                        "Model": model_name,
                                        "Cleaning Method": method,
                                        "Pearson r": score,
                                    }
                                )
                                completed_tasks.add(task_key)
                                pbar.update(1)

                                if len(records) % 10 == 0:
                                    pd.DataFrame(records).to_csv(
                                        self.config.RESULTS_CACHE_PATH, index=False
                                    )

                else:
                    train_idx, test_idx = self.splitter.get_indices(
                        df, X_ref, protocol
                    )

                    for emb_name, X_full in embeddings.items():
                        X_train_full, X_test = X_full[train_idx], X_full[test_idx]
                        y_train_full, y_test = y[train_idx], y[test_idx]

                        for method in cleaning_methods:
                            context_key = f"{protocol}_{emb_name}_train"
                            clean_mask = self.cleaner.get_clean_mask(
                                X_train_full, y_train_full, method, context_key
                            )
                            X_train_clean = X_train_full[clean_mask]
                            y_train_clean = y_train_full[clean_mask]

                            for model_name, model_obj in models.items():
                                task_key = (protocol, emb_name, model_name, method)
                                if task_key in completed_tasks:
                                    continue

                                try:
                                    if (
                                        len(y_train_clean) >= 2
                                        and len(y_test) >= 2
                                    ):
                                        model_obj.fit(
                                            X_train_clean, y_train_clean
                                        )
                                        preds = model_obj.predict(X_test)
                                        score = safe_pearsonr(y_test, preds)
                                    else:
                                        score = 0.0
                                except Exception:
                                    score = 0.0

                                records.append(
                                    {
                                        "Evaluation Protocol": protocol,
                                        "Representation": emb_name,
                                        "Model": model_name,
                                        "Cleaning Method": method,
                                        "Pearson r": score,
                                    }
                                )
                                completed_tasks.add(task_key)
                                pbar.update(1)

                                if len(records) % 10 == 0:
                                    pd.DataFrame(records).to_csv(
                                        self.config.RESULTS_CACHE_PATH, index=False
                                    )

        df_results = pd.DataFrame(records)
        os.makedirs(
            os.path.dirname(self.config.RESULTS_CACHE_PATH), exist_ok=True
        )
        df_results.to_csv(self.config.RESULTS_CACHE_PATH, index=False)
        print(
            f"\nBenchmark results saved to: {self.config.RESULTS_CACHE_PATH}"
        )
        return df_results


class AblationVisualizer:
    """Generates publication-ready master ablation heatmap."""

    @staticmethod
    def plot_heatmap(df_results: pd.DataFrame, output_path: str) -> None:
        """Plot and save wide heatmap matrix."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sns.set_theme(style="white", font="sans-serif")

        df_results["Pearson r"] = pd.to_numeric(
            df_results["Pearson r"], errors="coerce"
        )

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
        ax.set_xlabel(
            "Evaluation Protocol & Data Cleaning Strategy",
            fontsize=11,
            labelpad=10,
        )
        ax.set_ylabel("Representation & Regressor", fontsize=11, labelpad=10)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", fontsize=8
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Master heatmap exported to: {output_path}")


def main() -> None:
    """Execute complete pipeline."""
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