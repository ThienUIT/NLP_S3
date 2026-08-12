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
