"""Cohere reranker provider implementation (native v2 API)."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from esperanto.common_types import Model
from esperanto.common_types.reranker import RerankResponse, RerankResult
from esperanto.common_types.response import Usage

from .base import RerankerModel


@dataclass
class CohereRerankerModel(RerankerModel):
    """Cohere reranker provider using the native v2 rerank API."""

    def __post_init__(self):
        """Initialize Cohere reranker after dataclass initialization."""
        super().__post_init__()

        # Authentication
        self.api_key = self.api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cohere API key not found. Set the COHERE_API_KEY environment variable."
            )

        # API configuration (endpoints carry their own version segment)
        self.base_url = (self.base_url or "https://api.cohere.com").rstrip("/")

        # Initialize HTTP clients with configurable timeout
        self._create_http_clients()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers for Cohere API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_request_payload(
        self, query: str, documents: List[str], top_k: int, **kwargs
    ) -> Dict[str, Any]:
        """Build request payload for the Cohere rerank API.

        Cohere uses ``top_n`` (not ``top_k``) to cap the number of results.
        """
        payload = {
            "model": self.get_model_name(),
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }
        payload.update(kwargs)
        return payload

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle error responses from Cohere API."""
        try:
            error_data = response.json()
            error_message = error_data.get("message") or error_data.get(
                "error", "Unknown error"
            )
            raise RuntimeError(f"Cohere API error: {error_message}")
        except (KeyError, ValueError):
            raise RuntimeError(
                f"Cohere API error: {response.status_code} - {response.text}"
            )

    def _parse_response(
        self, response_data: Dict[str, Any], documents: List[str]
    ) -> RerankResponse:
        """Parse Cohere API response into the standardized format.

        Cohere returns results already sorted by relevance (descending) with
        ``index`` and ``relevance_score``. Documents are attached from the
        original list by index (Cohere v2 does not echo them back).
        """
        results = []
        raw_results = response_data.get("results", [])

        raw_scores = [result.get("relevance_score", 0.0) for result in raw_results]
        normalized_scores = self._normalize_scores(raw_scores)

        for i, result in enumerate(raw_results):
            index = result.get("index", i)
            document = documents[index] if index < len(documents) else ""

            results.append(
                RerankResult(
                    index=index,
                    document=document,
                    relevance_score=(
                        normalized_scores[i] if i < len(normalized_scores) else 0.0
                    ),
                )
            )

        usage = None
        billed = response_data.get("meta", {}).get("billed_units")
        if billed:
            usage = Usage(
                prompt_tokens=billed.get("input_tokens", 0),
                completion_tokens=billed.get("output_tokens", 0),
                total_tokens=billed.get("input_tokens", 0)
                + billed.get("output_tokens", 0),
            )

        return RerankResponse(
            results=results,
            model=self.get_model_name(),
            usage=usage,
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RerankResponse:
        """Rerank documents using the Cohere API."""
        query, documents, top_k = self._validate_inputs(query, documents, top_k)
        payload = self._build_request_payload(query, documents, top_k, **kwargs)

        try:
            response = self.client.post(
                f"{self.base_url}/v2/rerank",
                json=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                self._handle_error(response)

            return self._parse_response(response.json(), documents)

        except httpx.TimeoutException:
            raise RuntimeError("Request to Cohere API timed out")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error calling Cohere API: {str(e)}")

    async def arerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        **kwargs,
    ) -> RerankResponse:
        """Async rerank documents using the Cohere API."""
        query, documents, top_k = self._validate_inputs(query, documents, top_k)
        payload = self._build_request_payload(query, documents, top_k, **kwargs)

        try:
            response = await self.async_client.post(
                f"{self.base_url}/v2/rerank",
                json=payload,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                self._handle_error(response)

            return self._parse_response(response.json(), documents)

        except httpx.TimeoutException:
            raise RuntimeError("Request to Cohere API timed out")
        except httpx.RequestError as e:
            raise RuntimeError(f"Network error calling Cohere API: {str(e)}")

    def to_langchain(self):
        """Convert to a LangChain-compatible reranker."""
        try:
            from langchain_core.callbacks.manager import Callbacks
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError(
                "LangChain not installed. Install with: pip install langchain"
            )

        class LangChainCohereReranker:
            def __init__(self, cohere_reranker):
                self.cohere_reranker = cohere_reranker

            def compress_documents(
                self,
                documents: List[Document],
                query: str,
                callbacks: Optional[Callbacks] = None,
            ) -> List[Document]:
                """Compress documents using the Cohere reranker."""
                texts = [doc.page_content for doc in documents]
                rerank_response = self.cohere_reranker.rerank(query, texts)

                reranked_docs = []
                for result in rerank_response.results:
                    if result.index < len(documents):
                        original_doc = documents[result.index]
                        new_metadata = original_doc.metadata.copy()
                        new_metadata["relevance_score"] = result.relevance_score
                        reranked_docs.append(
                            Document(
                                page_content=original_doc.page_content,
                                metadata=new_metadata,
                            )
                        )
                return reranked_docs

        return LangChainCohereReranker(self)

    def _get_default_model(self) -> str:
        """Get the default Cohere reranker model."""
        return "rerank-v4.0-pro"

    @property
    def provider(self) -> str:
        """Provider name."""
        return "cohere"

    def _get_models(self) -> List[Model]:
        """Available Cohere reranker models."""
        try:
            response = self.client.get(
                f"{self.base_url}/v1/models",
                headers=self._get_headers(),
                params={"endpoint": "rerank"},
            )
            if response.status_code != 200:
                self._handle_error(response)
            models_data = response.json()
            return [
                Model(
                    id=model["name"],
                    owned_by="cohere",
                    context_window=model.get("context_length"),
                    type="reranker",
                )
                for model in models_data.get("models", [])
            ]
        except Exception:
            return [
                Model(
                    id="rerank-v4.0-pro",
                    owned_by="cohere",
                    context_window=4096,
                    type="reranker",
                ),
                Model(
                    id="rerank-v3.5",
                    owned_by="cohere",
                    context_window=4096,
                    type="reranker",
                ),
            ]
