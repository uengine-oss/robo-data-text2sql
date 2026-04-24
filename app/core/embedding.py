"""Embedding client for text vectorization"""
from typing import Any, List
from openai import AsyncOpenAI

from app.config import settings


class EmbeddingClient:
    """Client for generating text embeddings"""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        provider = (getattr(settings, "embedding_provider", "") or "").strip().lower()
        if provider not in {"openai", "openai_compatible"}:
            raise NotImplementedError(
                "embedding_provider={!r} is not supported yet (allowed: 'openai', "
                "'openai_compatible').".format(provider)
            )
        self.model = settings.embedding_model

    @staticmethod
    def _extract_embeddings(response: Any) -> List[Any]:
        data = getattr(response, "data", None)
        if data:
            return data

        err = getattr(response, "error", None)
        if err is not None:
            code = getattr(err, "code", None)
            message = getattr(err, "message", None)
            if message:
                raise RuntimeError(
                    f"Embedding API returned no data (code={code}, message={message})"
                )
        raise RuntimeError("Embedding API returned no data")
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        items = self._extract_embeddings(response)
        return items[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        items = self._extract_embeddings(response)
        return [item.embedding for item in items]
    
    @staticmethod
    def format_table_text(table_name: str, description: str = "", columns: List[str] = None) -> str:
        """Format table metadata for embedding"""
        parts = [f"Table: {table_name}"]
        if description:
            parts.append(f"Description: {description}")
        if columns:
            parts.append(f"Columns: {', '.join(columns)}")
        return " | ".join(parts)
    
    @staticmethod
    def format_column_text(column_name: str, table_name: str, dtype: str, description: str = "") -> str:
        """Format column metadata for embedding"""
        parts = [f"Column: {table_name}.{column_name}", f"Type: {dtype}"]
        if description:
            parts.append(f"Description: {description}")
        return " | ".join(parts)

