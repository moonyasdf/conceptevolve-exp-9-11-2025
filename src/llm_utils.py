# src/llm_utils.py
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import List, Optional, Type, TypeVar, Union

from google.genai import types
from omegaconf import DictConfig
from pydantic import BaseModel

from src.config import app_config

T = TypeVar("T", bound=BaseModel)


class LLMCache:
    """Simple cache layer for Gemini responses."""

    def __init__(self, cache_dir: str = ".cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(
        self, prompt: str, system_message: str, model: str, temperature: float
    ) -> str:
        content = f"{model}::{temperature:.2f}::{system_message}::{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(
        self, prompt: str, system_message: str, model: str, temperature: float
    ) -> Optional[str]:
        key = self._get_cache_key(prompt, system_message, model, temperature)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    return data.get("response")
            except Exception:
                return None
        return None

    def set(
        self,
        prompt: str,
        system_message: str,
        model: str,
        temperature: float,
        response: str,
    ) -> None:
        key = self._get_cache_key(prompt, system_message, model, temperature)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as handle:
                json.dump({"response": response}, handle, ensure_ascii=False)
        except Exception as exc:
            print(f"  ⚠️ Failed to write to cache: {exc}")


_cache = LLMCache()


def _handle_api_error() -> bool:
    """Handle Gemini API errors by prompting the operator."""
    while True:
        choice = input(
            "\n⚠️ Google API error.\n"
            "   1. Retry\n"
            "   2. Enter a new API key\n"
            "   3. Exit\n"
            "Choose (1, 2, or 3): "
        )
        if choice == "1":
            return True
        if choice == "2":
            app_config.api_key = None
            app_config.client = None
            app_config.get_client()
            return True
        if choice == "3":
            print("Exiting the program.")
            exit()
        print("Invalid option.")


def query_gemini_structured(
    prompt: str,
    system_message: str,
    response_schema: Type[T],
    model_cfg: DictConfig,
    use_cache: bool = True,
    max_retries: int = 3,
    temperature: Optional[float] = None,
) -> Optional[T]:
    """Call Gemini with structured output using the Hydra model configuration."""
    client = app_config.get_client()
    model_name = model_cfg.name
    
    # Use the temperature passed as argument, if not, use the evaluation one as fallback.
    generation_temperature = temperature if temperature is not None else model_cfg.temp_evaluation

    if use_cache:
        cached = _cache.get(prompt, system_message, model_name, generation_temperature)
        if cached:
            try:
                return response_schema.model_validate_json(cached)
            except Exception as exc:
                print(
                    f"  ⚠️ Cached response failed validation: {exc}. Requesting a fresh call."
                )

    for attempt in range(max_retries):
        try:
            generation_config = types.GenerateContentConfig(
                temperature=generation_temperature,
                system_instruction=system_message,
                response_mime_type="application/json",
                response_schema=response_schema,
            )

            if "2.5-pro" in model_name and model_cfg.use_thinking:
                generation_config.thinking_config = types.ThinkingConfig(
                    thinking_budget=model_cfg.thinking_budget
                )

            response = client.models.generate_content(
                model=f"models/{model_name}",
                contents=prompt,
                config=generation_config,
            )

            parsed_obj = response.parsed
            if not isinstance(parsed_obj, response_schema):
                raise TypeError(
                    "Structured response is not of type "
                    f"{response_schema.__name__}: {type(parsed_obj)}"
                )

            if use_cache:
                _cache.set(
                    prompt,
                    system_message,
                    model_name,
                    generation_temperature,
                    parsed_obj.model_dump_json(),
                )

            return parsed_obj

        except Exception as exc:
            print(
                f"  ❌ Structured API call failed (attempt {attempt + 1}/{max_retries}): {exc}"
            )
            if attempt >= max_retries - 1:
                if _handle_api_error():
                    return query_gemini_structured(
                        prompt,
                        system_message,
                        response_schema,
                        model_cfg,
                        use_cache,
                        max_retries,
                        temperature,
                    )
                return None
            time.sleep(2 ** attempt)
    return None


def query_gemini_unstructured(
    prompt: str,
    system_message: str,
    model_cfg: DictConfig,
    temperature: float,
    use_cache: bool = True,
    max_retries: int = 3,
) -> str:
    """Call Gemini for free-form text using the Hydra model configuration."""
    client = app_config.get_client()
    model_name = model_cfg.name

    if use_cache:
        cached = _cache.get(prompt, system_message, model_name, temperature)
        if cached:
            return cached

    for attempt in range(max_retries):
        try:
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_message,
            )

            if "2.5-pro" in model_name and model_cfg.use_thinking:
                generation_config.thinking_config = types.ThinkingConfig(
                    thinking_budget=model_cfg.thinking_budget
                )

            response = client.models.generate_content(
                model=f"models/{model_name}",
                contents=prompt,
                config=generation_config,
            )

            result_text = response.text

            if use_cache and result_text:
                _cache.set(prompt, system_message, model_name, temperature, result_text)

            return result_text

        except Exception as exc:
            print(
                f"  ❌ Unstructured API call failed (attempt {attempt + 1}/{max_retries}): {exc}"
            )
            if attempt >= max_retries - 1:
                if _handle_api_error():
                    return query_gemini_unstructured(
                        prompt,
                        system_message,
                        model_cfg,
                        temperature,
                        use_cache,
                        max_retries,
                    )
                return f"Unresolved error after retries: {exc}"
            time.sleep(2 ** attempt)
    return ""


class EmbeddingClient:
    """Wrapper around the Gemini embeddings API with sync and async helpers."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        if not self.model_name.startswith("models/"):
            self.model_name = f"models/{self.model_name}"

    def get_embedding(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        if not text:
            return [] if isinstance(text, str) else [[] for _ in text]

        client = app_config.get_client()
        is_single_item = isinstance(text, str)

        try:
            result = client.models.embed_content(model=self.model_name, contents=text)

            if is_single_item:
                return result.embeddings[0].values
            return [embedding.values for embedding in result.embeddings]

        except Exception as exc:
            print(f"  ❌ Failed to generate embedding: {exc}")
            if _handle_api_error():
                return self.get_embedding(text)
            return [] if is_single_item else [[] for _ in text]

    async def get_embedding_async(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embedding, text)
