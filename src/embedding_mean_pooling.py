"""
Embeddings con mean pooling para modelos encoder sin wrapper de
sentence-transformers (p.ej. BSC-LT/MrBERT-legal, BSC-LT/MrBERT-es).

Expone la misma interfaz que HuggingFaceEmbeddingsLC (embed_query,
embed_documents) para poder usarse como drop-in replacement en el
pipeline de ingestión.
"""

from typing import List

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


class MeanPoolingEmbeddings:
    """
    Extractor de embeddings basado en mean pooling (masked mean) sobre las
    últimas hidden states de un modelo encoder, con normalización L2
    opcional.

    Compatible con HuggingFaceEmbeddingsLC:
        - embed_query(text) -> List[float]
        - embed_documents(texts) -> List[List[float]]
    """

    def __init__(
        self,
        model_name: str = "BSC-LT/MrBERT-legal",
        show_progress: bool = True,
        batch_size: int = 32,
        max_length: int = 1024,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.show_progress = show_progress
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize

        # Selección de dispositivo (igual que HuggingFaceEmbeddingsLC)
        if torch.cuda.is_available():
            self.device = "cuda"
            print("CUDA is available. Using GPU for embeddings.")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("MPS is available. Using Apple Silicon GPU for embeddings.")
        else:
            self.device = "cpu"
            print("CUDA and MPS are not available. Using CPU for embeddings.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Algunos tokenizers SPM no definen pad_token explícitamente
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token or self.tokenizer.unk_token
            )

        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .to(last_hidden_state.dtype)
        )
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @torch.no_grad()
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self.model(**encoded)
        pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])

        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=1)

        return pooled.to(torch.float32).cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []

        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        batches = range(0, len(texts), self.batch_size)

        if self.show_progress:
            batches = tqdm(
                batches,
                total=n_batches,
                desc=f"Embedding ({self.model_name})",
            )

        for start in batches:
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._embed_batch(batch))

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]


if __name__ == "__main__":
    embedder = MeanPoolingEmbeddings(model_name="BSC-LT/MrBERT-legal")

    docs = [
        "La parte demandante presentó una demanda para reclamar daños y perjuicios.",
        "El Boletín Oficial del Estado publica hoy el real decreto sobre...",
    ]

    doc_embeddings = embedder.embed_documents(docs)
    query_embedding = embedder.embed_query(
        "¿Qué dice la ley sobre indemnización por daños?"
    )

    print(f"Documentos: {len(doc_embeddings)} embeddings de dimensión {len(doc_embeddings[0])}")
    print(f"Query: embedding de dimensión {len(query_embedding)}")