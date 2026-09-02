"""
Peptide Embedding Generation Pipeline with Subprocess Memory Isolation.

Features:
- Subprocess Execution: Each pLM runs in an isolated Python process. When finished,
  the OS reclaims 100% of allocated RAM/VRAM, completely eliminating memory leaks.
- Automatic Resume: Skips models whose .npy embedding file already exists on disk.
- Optimised Precision: FP32 for Ankh to prevent NaN overflow issues; FP16 on CUDA devices for others.
- Conservative Batching: Batch size 1 for ProtT5-XL to avoid VRAM spikes.
"""

from abc import ABC, abstractmethod
import argparse
import gc
import os
import subprocess
import sys
import traceback
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5EncoderModel,
)


class BaseEmbeddingExtractor(ABC):
    """Abstract base class for Protein Language Model embedding extractors."""

    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def extract_batch(self, batch_sequences: List[str]) -> np.ndarray:
        pass

    def extract_all(
        self, sequences: List[str], batch_size: int = 32
    ) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            self.load_model()

        all_embeddings = []
        for i in tqdm(
            range(0, len(sequences), batch_size),
            desc=f"Processing {self.model_name.split('/')[-1]}",
        ):
            batch = sequences[i : i + batch_size]
            embeddings = self.extract_batch(batch)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)


class ESMEmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for Meta's ESM-2 models."""

    def load_model(self) -> None:
        print(f"Loading ESM-2 model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            self.model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def extract_batch(self, batch_sequences: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            batch_sequences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state.float()
            attention_mask = inputs["attention_mask"]

            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

        return mean_pooled.cpu().numpy()


class ProtT5EmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for Rostlab's ProtT5-XL-UniRef50 model."""

    def load_model(self) -> None:
        print(f"Loading ProtT5 model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, do_lower_case=False
        )
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = T5EncoderModel.from_pretrained(
            self.model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def extract_batch(self, batch_sequences: List[str]) -> np.ndarray:
        spaced_sequences = [" ".join(list(seq)) for seq in batch_sequences]

        inputs = self.tokenizer(
            spaced_sequences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state.float()
            attention_mask = inputs["attention_mask"]

            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

        return mean_pooled.cpu().numpy()


class AnkhEmbeddingExtractor(BaseEmbeddingExtractor):
    """Embedding extractor for ElnaggarLab's Ankh model."""

    def load_model(self) -> None:
        print(f"Loading Ankh model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Enforce float32 to prevent float16 underflow/overflow NaNs
        dtype = torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def extract_batch(self, batch_sequences: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            batch_sequences, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            encoder_outputs = self.model.encoder(**inputs)
            last_hidden = encoder_outputs.last_hidden_state.float()
            attention_mask = inputs["attention_mask"]

            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

        return mean_pooled.cpu().numpy()


def find_sequence_column(df_columns: List[str]) -> Optional[str]:
    cleaned_cols = [str(col).strip().lower() for col in df_columns]
    candidates = [
        "sequence",
        "seq",
        "sekwencja",
        "peptide",
        "peptide_sequence",
        "peptidesequence",
    ]

    for cand in candidates:
        for idx, col in enumerate(cleaned_cols):
            if cand == col:
                return df_columns[idx]

    for idx, col in enumerate(cleaned_cols):
        if "seq" in col or "pept" in col:
            return df_columns[idx]

    return None


def load_and_preprocess_dataset(
    input_file: str, cleaned_csv_out: str
) -> pd.DataFrame:
    print(f"Loading dataset from: {input_file}")

    df = None
    try:
        df = pd.read_csv(input_file, sep=None, engine="python")
    except Exception:
        pass

    if df is None:
        try:
            df = pd.read_excel(input_file)
        except Exception:
            pass

    if df is None:
        try:
            df = pd.read_excel(input_file, engine="odf")
        except Exception:
            pass

    if df is None:
        raise ValueError(
            f"Unable to parse file '{input_file}'. Ensure CSV, XLSX, or ODS."
        )

    df.columns = [str(c).strip() for c in df.columns]

    seq_col = find_sequence_column(df.columns)
    if not seq_col:
        print("\n[ERROR] Available columns in file:")
        for col in df.columns:
            print(f"  - '{col}'")
        raise KeyError("Could not identify a 'sequence' column in input file.")

    print(f"Detected sequence column: '{seq_col}'")
    df = df.rename(columns={seq_col: "sequence"})

    df = df.dropna(subset=["sequence"]).copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

    df = df.reset_index(drop=True)
    df.to_csv(cleaned_csv_out, index=False)

    print(f"Dataset preprocessed. Valid peptides count: {len(df)}")
    print(f"Cleaned dataset saved to: {cleaned_csv_out}")

    return df


MODELS_CONFIG = [
    # {
    #     "class": ESMEmbeddingExtractor,
    #     "name": "facebook/esm2_t6_8M_UR50D",
    #     "out_file": "esm2_8m_embeddings.npy",
    #     "batch_size": 32,
    # },
    # {
    #     "class": ESMEmbeddingExtractor,
    #     "name": "facebook/esm2_t12_35M_UR50D",
    #     "out_file": "esm2_35m_embeddings.npy",
    #     "batch_size": 16,
    # },
    # {
    #     "class": ESMEmbeddingExtractor,
    #     "name": "facebook/esm2_t33_650M_UR50D",
    #     "out_file": "esm2_650m_embeddings.npy",
    #     "batch_size": 2,
    # },
    # {
    #     "class": ProtT5EmbeddingExtractor,
    #     "name": "Rostlab/prot_t5_xl_uniref50",
    #     "out_file": "prott5_embeddings.npy",
    #     "batch_size": 1,
    # },
    {
        "class": AnkhEmbeddingExtractor,
        "name": "ElnaggarLab/ankh-base",
        "out_file": "ankh_base_embeddings.npy",
        "batch_size": 4,
    },
]


def run_single_model(model_idx: int, base_dir: str):
    """Runs embedding extraction for a single model index in isolation."""
    output_dir = os.path.join(base_dir, "embeddings_data")
    cleaned_csv = os.path.join(output_dir, "cleaned_peptides.csv")

    config = MODELS_CONFIG[model_idx]
    out_path = os.path.join(output_dir, config["out_file"])

    if os.path.exists(out_path):
        print(f"Skipping {config['name']} — Already generated: {out_path}")
        return

    print("\n" + "=" * 60)
    print(f"Running isolated extraction process for: {config['name']}")

    if not os.path.exists(cleaned_csv):
        raise FileNotFoundError(f"Cleaned CSV dataset not found at: {cleaned_csv}")

    df = pd.read_csv(cleaned_csv)
    sequences = df["sequence"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device.type.upper()}")

    try:
        extractor = config["class"](model_name=config["name"], device=device)
        embeddings = extractor.extract_all(
            sequences=sequences, batch_size=config["batch_size"]
        )
        np.save(out_path, embeddings)
        print(
            f"Successfully saved embeddings to {out_path} "
            f"| Shape: {embeddings.shape}"
        )
    except Exception as err:
        print(f"\n[ERROR] Failed processing model: {config['name']}")
        print(f"Error details: {err}")
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Extract Peptide Embeddings.")
    parser.add_argument(
        "--model_idx",
        type=int,
        default=-1,
        help="Index of specific model to process (-1 for runner mode)",
    )
    args = parser.parse_args()

    base_dir = "/home/marta/Pulpit/ACEpIC"

    # Worker mode: execute only one model in this clean process
    if args.model_idx >= 0:
        run_single_model(args.model_idx, base_dir)
        sys.exit(0)

    # Master runner mode
    input_file = os.path.join(
        base_dir, "dataset/final/experimental_dataset.csv"
    )
    output_dir = os.path.join(base_dir, "embeddings_data")
    cleaned_csv = os.path.join(output_dir, "cleaned_peptides.csv")

    os.makedirs(output_dir, exist_ok=True)

    # Preprocess dataset once
    try:
        load_and_preprocess_dataset(input_file, cleaned_csv)
    except Exception as err:
        print(f"Failed to load dataset: {err}")
        traceback.print_exc()
        sys.exit(1)

    script_path = os.path.abspath(__file__)

    # Sequentially launch a separate Python process for each model
    for idx, config in enumerate(MODELS_CONFIG):
        out_path = os.path.join(output_dir, config["out_file"])
        if os.path.exists(out_path):
            print(f"\n[SKIP] {config['out_file']} already exists in {output_dir}")
            continue

        cmd = [sys.executable, script_path, "--model_idx", str(idx)]
        print(f"\nLaunch Subprocess -> Model #{idx}: {config['name']}")

        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[WARNING] Subprocess for model #{idx} exited with code {res.returncode}")

    print("\n" + "=" * 60)
    print(f"All processing completed! Check directory: {output_dir}")


if __name__ == "__main__":
    main()