import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

# Resolve repository root directory dynamically (ACEpIC/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config:
    """Paths and default settings for model training and live inference."""

    DATASET_PATH: Path = PROJECT_ROOT / "embeddings_data" / "cleaned_peptides.csv"
    PROTT5_EMBEDDINGS_PATH: Path = (
        PROJECT_ROOT / "embeddings_data" / "prott5_embeddings.npy"
    )
    ESM2_650M_EMBEDDINGS_PATH: Path = (
        PROJECT_ROOT / "embeddings_data" / "esm2_650m_embeddings.npy"
    )
    CLEANING_CUTOFF_PERCENTILE: float = 80.0  # Retain bottom 80% (remove 20% highest residuals)
    RIDGE_ALPHA: float = 1.0
    PROTT5_MODEL_NAME: str = "Rostlab/prot_t5_xl_half_uniref50-enc"
    ESM2_MODEL_NAME: str = "facebook/esm2_t33_650M_UR50D"
    TARGET_PEPTIDES: str = (
        "IAVL IVVL IVPL IQL IDL IDF VTK VSK ISL VAK ASF VPK APK IVK IPK"
    )


class FusedRidgePipeline:
    """Handles dataset loading, residual data cleaning, and Ridge model training."""

    def __init__(self, config: Config = Config()):
        self.config = config
        self.model = Ridge(alpha=self.config.RIDGE_ALPHA)

    def load_dataset_and_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load target labels and combine ProtT5 + ESM2 feature representations."""
        if not self.config.DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset CSV file missing at: {self.config.DATASET_PATH}"
            )
        if not self.config.PROTT5_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"ProtT5 embeddings missing at: {self.config.PROTT5_EMBEDDINGS_PATH}"
            )
        if not self.config.ESM2_650M_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"ESM2 embeddings missing at: {self.config.ESM2_650M_EMBEDDINGS_PATH}"
            )

        df = pd.read_csv(self.config.DATASET_PATH)
        if "pIC50" not in df.columns:
            raise KeyError("Column 'pIC50' is missing from the dataset CSV.")

        y = df["pIC50"].values
        x_prott5 = np.load(self.config.PROTT5_EMBEDDINGS_PATH)
        x_esm2 = np.load(self.config.ESM2_650M_EMBEDDINGS_PATH)

        X_fused = np.hstack((x_prott5, x_esm2))
        return X_fused, y

    def fit_with_residual_cleaning(self) -> None:
        """Fit Ridge model after removing 20% of samples with highest residual errors."""
        X, y = self.load_dataset_and_embeddings()

        # Step 1: Initial fit to evaluate sample residuals
        initial_ridge = Ridge(alpha=self.config.RIDGE_ALPHA)
        initial_ridge.fit(X, y)
        residuals = np.abs(y - initial_ridge.predict(X))

        # Step 2: Apply 20% residual threshold cleaning
        cutoff_threshold = np.percentile(
            residuals, self.config.CLEANING_CUTOFF_PERCENTILE
        )
        clean_mask = residuals <= cutoff_threshold

        retained_count = int(np.sum(clean_mask))
        removed_count = len(y) - retained_count
        print(
            f"Training final Ridge model on {retained_count} clean samples "
            f"(removed {removed_count} noisy samples via 20% residual cutoff)..."
        )

        # Step 3: Train final model on cleaned subset
        self.model.fit(X[clean_mask], y[clean_mask])

    def predict(self, fused_embedding: np.ndarray) -> float:
        """Predict continuous pIC50 target value from input feature vector."""
        return float(self.model.predict(fused_embedding)[0])


class OnlinePeptideEmbedder:
    """Extracts live representations from ProtT5-XL and ESM-2 650M models on CPU."""

    def __init__(self, config: Config = Config()):
        self.config = config
        # Enforce CPU execution to prevent GPU CUDA Out-Of-Memory (OOM) errors
        self.device = torch.device("cpu")
        print(f"Initializing pLM feature extractors on device: {self.device}")

        print(f"Loading ProtT5 encoder ({self.config.PROTT5_MODEL_NAME})...")
        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            self.config.PROTT5_MODEL_NAME, do_lower_case=False
        )
        self.t5_model = T5EncoderModel.from_pretrained(
            self.config.PROTT5_MODEL_NAME
        ).to(self.device)
        self.t5_model.eval()

        print(f"Loading ESM-2 model ({self.config.ESM2_MODEL_NAME})...")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            self.config.ESM2_MODEL_NAME
        )
        self.esm_model = AutoModel.from_pretrained(
            self.config.ESM2_MODEL_NAME
        ).to(self.device)
        self.esm_model.eval()

    def get_prott5_embedding(self, sequence: str) -> np.ndarray:
        """Generate mean-pooled ProtT5 representation for single sequence."""
        formatted_seq = " ".join(list(sequence.upper()))
        inputs = self.t5_tokenizer(
            formatted_seq, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.t5_model(**inputs)
            embedding = (
                outputs.last_hidden_state[0, :-1, :].mean(dim=0).cpu().numpy()
            )
        return embedding

    def get_esm2_embedding(self, sequence: str) -> np.ndarray:
        """Generate mean-pooled ESM-2 650M representation for single sequence."""
        inputs = self.esm_tokenizer(
            sequence.upper(), return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            embedding = (
                outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy()
            )
        return embedding

    def get_fused_embedding(self, sequence: str) -> np.ndarray:
        """Extract and concatenate ProtT5 and ESM-2 embeddings into single array."""
        emb_t5 = self.get_prott5_embedding(sequence)
        emb_esm = self.get_esm2_embedding(sequence)
        fused_vector = np.hstack((emb_t5, emb_esm))
        return fused_vector.reshape(1, -1)


def is_valid_peptide_sequence(sequence: str) -> bool:
    """Validate whether input string consists of valid canonical amino acid codes."""
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    clean_str = sequence.strip().upper()
    return len(clean_str) > 0 and set(clean_str).issubset(valid_amino_acids)


def parse_input_sequences(user_input: str) -> List[str]:
    """Extract valid peptide sequences from raw user string or file path."""
    file_path = Path(user_input)
    if file_path.is_file():
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        raw_seqs = [
            line.strip() for line in lines if not line.startswith(">") and line.strip()
        ]
    else:
        clean_input = user_input.replace(",", " ")
        raw_seqs = clean_input.split()

    return [s.strip().upper() for s in raw_seqs if is_valid_peptide_sequence(s)]


def predict_batch_sequences(
    sequences: List[str],
    ridge_pipeline: FusedRidgePipeline,
    embedder: OnlinePeptideEmbedder,
) -> pd.DataFrame:
    """Predict pIC50 and IC50 (µM) values for a list of peptide sequences."""
    print(f"\nProcessing {len(sequences)} sequence(s)...")
    results = []

    for seq in sequences:
        fused_emb = embedder.get_fused_embedding(seq)
        pred_pic50 = ridge_pipeline.predict(fused_emb)
        pred_ic50_um = 10 ** (6 - pred_pic50)
        results.append(
            {"Peptide": seq, "pIC50": pred_pic50, "IC50 (µM)": pred_ic50_um}
        )

    return pd.DataFrame(results)


def main() -> None:
    """Execute IC50 prediction pipeline with pre-defined peptide list and interactive prompt."""
    try:
        config = Config()

        print("=== ACEpIC: Fused (ProtT5 + ESM2) Ridge IC50 Predictor ===")
        ridge_pipeline = FusedRidgePipeline(config)
        ridge_pipeline.fit_with_residual_cleaning()

        embedder = OnlinePeptideEmbedder(config)

        # 1. Run batch prediction for target peptides automatically
        target_seqs = parse_input_sequences(config.TARGET_PEPTIDES)
        print("\n" + "=" * 60)
        print(" PREDICTIONS FOR TARGET PEPTIDES")
        print("=" * 60)
        df_targets = predict_batch_sequences(
            target_seqs, ridge_pipeline, embedder
        )
        print("\n" + df_targets.to_string(index=False))
        print("=" * 60 + "\n")

        # 2. Interactive Loop
        print(" Interactive Prediction Mode")
        print(" Enter custom peptide(s), list, or file path. Enter 'q' to quit.")
        print("-" * 60 + "\n")

        while True:
            user_input = input("Enter peptide(s) or file path: ").strip()

            if user_input.lower() in ["q", "exit", "quit"]:
                print("Exiting interactive prediction module.")
                break

            if not user_input:
                continue

            sequences = parse_input_sequences(user_input)

            if not sequences:
                print(
                    "Error: No valid peptide sequences found! "
                    "Ensure input contains standard single-letter codes (A-Z).\n"
                )
                continue

            df_res = predict_batch_sequences(sequences, ridge_pipeline, embedder)
            print("\n" + df_res.to_string(index=False))
            print("-" * 60 + "\n")

    except Exception as error:
        print(f"Execution failed: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()