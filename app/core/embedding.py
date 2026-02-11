"""Embedding client for text vectorization"""
from typing import List
from openai import AsyncOpenAI

from app.config import settings


class EmbeddingClient:
    """Client for generating text embeddings"""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        provider = (getattr(settings, "embedding_provider", "") or "").strip().lower()
        if provider not in {"openai"}:
            raise NotImplementedError(
                f"embedding_provider={provider!r} is not supported yet (only 'openai' is supported)."
            )
        self.model = settings.embedding_model
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float"
        )
        return [item.embedding for item in response.data]
    
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

