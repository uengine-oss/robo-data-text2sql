"""Embedding client for text vectorization.

spec 007 T1 (FR-020~FR-023) — 임베딩 제공자 심.

**왜 고쳤는가.**
이전 판은 생성자에서 `embedding_provider != "openai"` 를 `NotImplementedError` 로 막았다.
그 두 줄이 "로컬 임베딩" 선택지 전체를 구조적으로 불가능하게 만들고 있었다
(spec CR-5b 결과 2). 설정 화면을 아무리 잘 만들어도 여기서 예외가 나면 화면에만 존재하는
선택지가 된다. 그래서 **제약을 제거하고**, 어떤 제공자로 만들어졌는지는 막는 대신
`fingerprint()` 로 **기록**한다(T2).

**왜 provider/model 을 클라이언트가 들고 있는가.**
벡터 옆에 지문을 남기려면(FR-050) "이 벡터를 누가 만들었는가"를 벡터를 만든 그 객체가
알고 있어야 한다. 지문을 쓸 때 전역 설정을 다시 읽는 방식은 색인 도중 설정이 바뀌면
**거짓 지문**을 남긴다.
"""
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI

from app.config import settings


class EmbeddingClient:
    """Client for generating text embeddings.

    provider/model/base_url 은 생성 시점에 고정된다 — 색인이 도는 도중 전역 설정이 바뀌어도
    이미 시작한 배치는 자기가 시작한 설정으로 끝나고, 그 설정이 그대로 지문으로 남는다.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.client = client
        # provider 를 여기서 검사하지 않는다 — 검사는 팩토리(llm_factory)가 한다.
        # 이 클래스는 "어떤 OpenAI 호환 엔드포인트든 왕복한다" 만 책임진다.
        self.provider = (
            provider if provider is not None else (getattr(settings, "embedding_provider", "") or "")
        ).strip().lower()
        self.model = (model if model is not None else (settings.embedding_model or "")).strip()
        self.base_url = ((base_url or "").strip() or None)

    # ------------------------------------------------------------------
    # 지문 (spec 007 T2 / FR-050)
    # ------------------------------------------------------------------
    def fingerprint(self, *, dimension: Optional[int] = None) -> Dict[str, Any]:
        """이 클라이언트가 만든 벡터에 함께 기록할 지문.

        dimension 은 **실측값**을 넘기는 것이 원칙이다. 설정값(`EMBEDDING_DIMENSION`)은
        설정이지 데이터에 붙은 사실이 아니다 (spec CR-5c).
        """
        return {
            "embedding_provider": self.provider,
            "embedding_model": self.model,
            "embedding_dimension": int(dimension) if dimension else int(settings.embedding_dimension),
        }


    def _extra(self) -> Dict[str, Any]:
        """`dimensions` 를 보낼지.

        **왜 옵션인가.** 기존 Neo4j 벡터 인덱스는 1536 차원인데 로컬 모델은 대개 다른
        차원을 낸다(실측: `qwen3-embedding` = 4096). MRL 계열 모델과 OpenAI
        `text-embedding-3-*` 는 `dimensions` 로 잘라 줄 수 있어서, 이 한 줄이 있으면
        **인덱스를 다시 만들지 않고** 로컬 엔드포인트로 갈아탈 수 있다.
        지원하지 않는 서버에 보내면 400 이 나므로 기본은 끔 — 켜는 것은 사용자 선택이다.
        """
        if not bool(getattr(settings, "embedding_send_dimensions", False)):
            return {}
        dim = int(getattr(settings, "embedding_dimension", 0) or 0)
        return {"dimensions": dim} if dim > 0 else {}

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
            **self._extra(),
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
            **self._extra(),
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

