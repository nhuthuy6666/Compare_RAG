from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError, OpenAI
from pydantic import Field, PrivateAttr

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.types import PydanticProgramMode


class OpenAICompatibleEmbedding(BaseEmbedding):
    model_name: str = Field(description="Embedding model name exposed by an OpenAI-compatible endpoint.")
    api_base: str | None = Field(default=None, exclude=True)
    api_key: str | None = Field(default=None, exclude=True)
    dimensions: int | None = Field(default=None)
    timeout: float = Field(default=120.0, exclude=True)
    retry_attempts: int = Field(default=5, exclude=True)
    retry_delay: float = Field(default=5.0, exclude=True)
    default_headers: dict[str, str] | None = Field(default=None, exclude=True)

    _client: OpenAI = PrivateAttr()
    _async_client: AsyncOpenAI = PrivateAttr()

    # Khởi tạo wrapper embedding và dựng sẵn sync/async client.
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._build_clients()

    # Tạo lại OpenAI-compatible clients khi khởi động hoặc sau một lần retry.
    def _build_clients(self) -> None:
        client_kwargs = {
            "api_key": self.api_key or "ollama",
            "base_url": self.api_base,
            "timeout": self.timeout,
            "default_headers": self.default_headers,
        }
        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)

    # Chuẩn hóa payload request embedding cho cả single input và batch input.
    def _embedding_kwargs(self, inputs: str | list[str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": inputs,
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        return kwargs

    # Lấy embedding cho query bằng cùng logic như text thường.
    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    # Lấy embedding bất đồng bộ cho query.
    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await self._aget_text_embedding(query)

    # Lấy embedding đồng bộ cho một đoạn text.
    def _get_text_embedding(self, text: str) -> list[float]:
        response = self._request_embeddings_sync(text)
        return list(response.data[0].embedding)

    # Lấy embedding bất đồng bộ cho một đoạn text.
    async def _aget_text_embedding(self, text: str) -> list[float]:
        response = await self._request_embeddings_async(text)
        return list(response.data[0].embedding)

    # Lấy embedding cho cả batch; nếu batch lỗi thì tự chia nhỏ để tăng độ bền ingest.
    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._request_embeddings_sync(texts)
            return [list(item.embedding) for item in response.data]
        except (APITimeoutError, APIConnectionError, httpx.TimeoutException, httpx.ConnectError, BadRequestError) as exc:
            if isinstance(exc, BadRequestError) and not _is_retryable_embedding_error(exc):
                raise
            if len(texts) <= 1:
                raise
            midpoint = len(texts) // 2
            return self._get_text_embeddings(texts[:midpoint]) + self._get_text_embeddings(texts[midpoint:])

    # Phiên bản async của batch embedding với cùng chiến lược chia nhỏ batch khi gặp lỗi.
    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._request_embeddings_async(texts)
            return [list(item.embedding) for item in response.data]
        except (APITimeoutError, APIConnectionError, httpx.TimeoutException, httpx.ConnectError, BadRequestError) as exc:
            if isinstance(exc, BadRequestError) and not _is_retryable_embedding_error(exc):
                raise
            if len(texts) <= 1:
                raise
            midpoint = len(texts) // 2
            left = await self._aget_text_embeddings(texts[:midpoint])
            right = await self._aget_text_embeddings(texts[midpoint:])
            return left + right

    # Gửi request embedding đồng bộ với retry và tái tạo client khi cần.
    def _request_embeddings_sync(self, inputs: str | list[str]):
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return self._client.embeddings.create(**self._embedding_kwargs(inputs))
            except (APITimeoutError, APIConnectionError, httpx.TimeoutException, httpx.ConnectError, BadRequestError) as exc:
                if isinstance(exc, BadRequestError) and not _is_retryable_embedding_error(exc):
                    raise
                if attempt == self.retry_attempts:
                    print(
                        "Embedding request failed after all retries. "
                        "Please check whether the OpenAI-compatible endpoint is still running."
                    )
                    raise
                print(
                    f"Embedding request failed ({type(exc).__name__}), "
                    f"retrying {attempt}/{self.retry_attempts - 1} in {self.retry_delay:.1f}s..."
                )
                time.sleep(self.retry_delay)
                self._build_clients()

    # Gửi request embedding bất đồng bộ với retry và tái tạo client khi cần.
    async def _request_embeddings_async(self, inputs: str | list[str]):
        for attempt in range(1, self.retry_attempts + 1):
            try:
                return await self._async_client.embeddings.create(**self._embedding_kwargs(inputs))
            except (APITimeoutError, APIConnectionError, httpx.TimeoutException, httpx.ConnectError, BadRequestError) as exc:
                if isinstance(exc, BadRequestError) and not _is_retryable_embedding_error(exc):
                    raise
                if attempt == self.retry_attempts:
                    print(
                        "Embedding request failed after all retries. "
                        "Please check whether the OpenAI-compatible endpoint is still running."
                    )
                    raise
                print(
                    f"Embedding request failed ({type(exc).__name__}), "
                    f"retrying {attempt}/{self.retry_attempts - 1} in {self.retry_delay:.1f}s..."
                )
                await asyncio.sleep(self.retry_delay)
                self._build_clients()


# Nhận diện các lỗi embedding có thể retry an toàn, nhất là lỗi mạng/timeout từ Ollama.
def _is_retryable_embedding_error(exc: Exception) -> bool:
    message = str(exc).lower()
    retryable_fragments = (
        "health resp",
        "forcibly closed by the remote host",
        "connection was forcibly closed",
        "wsarecv",
        "connection reset",
        "actively refused it",
        "connectex",
        "read tcp",
        "dial tcp",
    )
    return any(fragment in message for fragment in retryable_fragments)


class OpenAICompatibleLLM(CustomLLM):
    model_name: str = Field(description="LLM model name exposed by an OpenAI-compatible endpoint.")
    api_base: str | None = Field(default=None, exclude=True)
    api_key: str | None = Field(default=None, exclude=True)
    context_window: int = Field(default=32768)
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.1)
    timeout: float = Field(default=120.0, exclude=True)
    default_headers: dict[str, str] | None = Field(default=None, exclude=True)

    _client: OpenAI = PrivateAttr()

    # Khởi tạo wrapper LLM để LlamaIndex gọi qua API chat completion chuẩn OpenAI-compatible.
    def __init__(
        self,
        model_name: str,
        api_base: str | None = None,
        api_key: str | None = None,
        context_window: int = 32768,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        timeout: float = 120.0,
        default_headers: dict[str, str] | None = None,
        callback_manager: CallbackManager | None = None,
        system_prompt: str | None = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            context_window=context_window,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            default_headers=default_headers,
            callback_manager=callback_manager or CallbackManager([]),
            system_prompt=system_prompt,
            pydantic_program_mode=pydantic_program_mode,
            **kwargs,
        )
        self._client = OpenAI(
            api_key=self.api_key or "ollama",
            base_url=self.api_base,
            timeout=self.timeout,
            default_headers=self.default_headers,
        )

    # Khai báo metadata để LlamaIndex biết context window và output size của model.
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            is_chat_model=False,
            model_name=self.model_name,
        )

    # Sinh completion đồng bộ từ prompt đơn.
    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        text = response.choices[0].message.content or ""
        return CompletionResponse(text=text, raw=response.model_dump())

    # Giả lập streaming bằng cách trả về một completion duy nhất theo interface của LlamaIndex.
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        # Bọc completion đơn thành generator để khớp interface streaming của LlamaIndex.
        def gen() -> CompletionResponseGen:
            response = self.complete(prompt, formatted=formatted, **kwargs)
            yield CompletionResponse(text=response.text, delta=response.text, raw=response.raw)

        return gen()
