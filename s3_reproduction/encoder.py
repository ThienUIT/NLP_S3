from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


@dataclass
class CafeBERTEncoder:
    model_name: str = "uitnlp/CafeBERT"
    batch_size: int = 8
    max_length: int = 256
    device: str = "auto"

    def __post_init__(self) -> None:
        self.device = (
            "cuda" if self.device == "auto" and torch.cuda.is_available() else
            "cpu" if self.device == "auto" else self.device
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except (OSError, requests.RequestException):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model_kwargs = {"dtype": dtype, "add_pooling_layer": False}
        try:
            model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
        except (OSError, requests.RequestException):
            model = AutoModel.from_pretrained(self.model_name, local_files_only=True, **model_kwargs)
        self.model = model.to(self.device).eval()

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        expanded = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)

    def encode(self, texts: list[str], description: str = "Encoding") -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc=description):
            batch = texts[start : start + self.batch_size]
            tokens = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**tokens).last_hidden_state
                pooled = self._mean_pool(hidden, tokens["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            batches.append(pooled.cpu().numpy())
        return np.concatenate(batches, axis=0)


def default_prefix(model_name: str) -> str:
    """Only the E5 family expects a "query: "/"passage: " prefix; applying it
    to a model that wasn't trained with one (bge-m3, mpnet, Vietnamese SimCSE
    models) would just glue literal text onto every input and hurt quality.
    """
    return "query: " if "e5" in model_name.lower() else ""


@dataclass
class SentenceTransformerEncoder:
    """Wraps a real sentence-transformers model -- built-in pooling and
    normalization, no hand-rolled masked-mean-pooling like CafeBERTEncoder
    needs. CafeBERT is a masked-LM continued-pretrained on Vietnamese, not
    trained as a sentence embedder (see REPRODUCE.md's own adaptation note);
    this class is for encoders that ARE. model_name can be any sentence-
    transformers-compatible checkpoint (multilingual-e5-*, BAAI/bge-m3,
    paraphrase-multilingual-mpnet-base-v2, Vietnamese-specific SimCSE/bi-
    encoders...) -- default is multilingual-e5-base, the same encoder family
    the paper benchmarks (§4.2 uses E5-large-v2), swapped for its
    multilingual variant so it can encode Vietnamese at all.
    """

    model_name: str = "intfloat/multilingual-e5-base"
    batch_size: int = 32
    max_length: int = 256
    device: str = "auto"
    prefix: str | None = None  # None = infer from model_name (see default_prefix)

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = (
            "cuda" if self.device == "auto" and torch.cuda.is_available() else
            "cpu" if self.device == "auto" else self.device
        )
        if self.prefix is None:
            self.prefix = default_prefix(self.model_name)
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.max_seq_length = self.max_length

    def encode(self, texts: list[str], description: str = "Encoding") -> np.ndarray:
        prefixed = [f"{self.prefix}{text}" for text in texts] if self.prefix else list(texts)
        return self.model.encode(
            prefixed, batch_size=self.batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True,
        )


ENCODERS = {"cafebert": CafeBERTEncoder, "e5": SentenceTransformerEncoder}
