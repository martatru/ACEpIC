"""
Peptide Embedding Generation Pipeline with Optimized Memory Management.

Features:
- FP16 (half-precision) loading on CUDA.
- Explicit PyTorch CUDA cache clearing & Garbage Collection.
- Automatic resume (skips already generated .npy files).
- Reduced batch sizes for large pLMs.
"""

from abc import ABC, abstractmethod
import gc
import os
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
            last_hidden = outputs.last_hidden_state.float()  # convert to float32 for pooling
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
        
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
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


def clear_memory():
    """Forces garbage collection and flushes PyTorch CUDA memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    base_dir = "/home/marta/Pulpit/ACEpIC"
    input_file = os.path.join(
        base_dir, "dataset/final/experimental_dataset.csv"
    )
    output_dir = os.path.join(base_dir, "embeddings_data")
    cleaned_csv = os.path.join(output_dir, "cleaned_peptides.csv")

    os.makedirs(output_dir, exist_ok=True)

    try:
        df = load_and_preprocess_dataset(input_file, cleaned_csv)
    except Exception as err:
        print(f"Failed to load dataset: {err}")
        traceback.print_exc()
        sys.exit(1)

    sequences = df["sequence"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device.type.upper()}")

    # Conservative batch sizes to avoid Memory Shortage
    models_to_run = [
        {
            "class": ESMEmbeddingExtractor,
            "name": "facebook/esm2_t6_8M_UR50D",
            "out_file": "esm2_8m_embeddings.npy",
            "batch_size": 32,
        },
        {
            "class": ESMEmbeddingExtractor,
            "name": "facebook/esm2_t12_35M_UR50D",
            "out_file": "esm2_35m_embeddings.npy",
            "batch_size": 16,
        },
        {
            "class": ESMEmbeddingExtractor,
            "name": "facebook/esm2_t33_650M_UR50D",
            "out_file": "esm2_650m_embeddings.npy",
            "batch_size": 4,
        },
        {
            "class": ProtT5EmbeddingExtractor,
            "name": "Rostlab/prot_t5_xl_uniref50",
            "out_file": "prott5_embeddings.npy",
            "batch_size": 2,
        },
        {
            "class": AnkhEmbeddingExtractor,
            "name": "ElnaggarLab/ankh-base",
            "out_file": "ankh_base_embeddings.npy",
            "batch_size": 8,
        },
    ]

    for config in models_to_run:
        print("\n" + "=" * 60)
        out_path = os.path.join(output_dir, config["out_file"])

        # Resume mechanism: Skip if already computed
        if os.path.exists(out_path):
            print(f"Skipping {config['name']} — Output file already exists: {out_path}")
            continue

        try:
            extractor = config["class"](
                model_name=config["name"], device=device
            )
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
        finally:
            # Force cleanup after each model
            if 'extractor' in locals():
                del extractor
            clear_memory()

    print("\n" + "=" * 60)
    print(f"Execution finished. Check directory: {output_dir}")


if __name__ == "__main__":
    main()