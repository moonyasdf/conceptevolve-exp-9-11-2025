# src/llm_utils.py
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any, List, Optional, Type, TypeVar, Union

from google.genai import types
from omegaconf import DictConfig
from pydantic import BaseModel

from src.config import app_config

T = TypeVar("T", bound=BaseModel)


class LLMCache:
    """Simple cache layer for LLM responses."""

    def __init__(self, cache_dir: str = ".cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(
        self, prompt: str, system_message: str, model: str, temperature: float, provider: str
    ) -> str:
        content = f"{provider}::{model}::{temperature:.2f}::{system_message}::{prompt}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(
        self, prompt: str, system_message: str, model: str, temperature: float, provider: str = "gemini"
    ) -> Optional[str]:
        key = self._get_cache_key(prompt, system_message, model, temperature, provider)
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
        provider: str = "gemini",
    ) -> None:
        key = self._get_cache_key(prompt, system_message, model, temperature, provider)
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as handle:
                json.dump({"response": response}, handle, ensure_ascii=False)
        except Exception as exc:
            print(f"  ⚠️ Failed to write to cache: {exc}")


_cache = LLMCache()


def _cfg_attr(cfg: Optional[DictConfig], attr: str, default: Any) -> Any:
    if cfg is None:
        return default
    value = getattr(cfg, attr, None)
    return default if value is None else value


def _handle_api_error(provider: str = "gemini") -> bool:
    """Handle LLM API errors by prompting the operator."""
    provider_name = "Google" if provider == "gemini" else "OpenAI"
    while True:
        choice = input(
            f"\n⚠️ {provider_name} API error.\n"
            "   1. Retry\n"
            "   2. Enter a new API key\n"
            "   3. Exit\n"
            "Choose (1, 2, or 3): "
        )
        if choice == "1":
            return True
        if choice == "2":
            if provider == "gemini":
                app_config.gemini_api_key = None
                app_config.gemini_client = None
                app_config.get_client("gemini")
            else:
                app_config.openai_api_key = None
                app_config.openai_client = None
                app_config.get_client("openai")
            return True
        if choice == "3":
            print("Exiting the program.")
            exit()
        print("Invalid option.")


def _query_gemini_structured(
    prompt: str,
    system_message: str,
    response_schema: Type[T],
    model_cfg: DictConfig,
    use_cache: bool = True,
    max_retries: int = 3,
    temperature: Optional[float] = None,
) -> Optional[T]:
    """Call Gemini with structured output using the Hydra model configuration."""
    client = app_config.get_client("gemini")
    model_name = getattr(model_cfg, "name", app_config.model_config.gemini.model_name)

    temp_eval = _cfg_attr(model_cfg, "temp_evaluation", app_config.model_config.temp_evaluation)
    generation_temperature = temperature if temperature is not None else temp_eval

    if use_cache:
        cached = _cache.get(prompt, system_message, model_name, generation_temperature, "gemini")
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

            use_thinking = _cfg_attr(model_cfg, "use_thinking", app_config.model_config.use_thinking)
            if "2.5-pro" in model_name and use_thinking:
                thinking_budget = _cfg_attr(model_cfg, "thinking_budget", app_config.model_config.thinking_budget)
                generation_config.thinking_config = types.ThinkingConfig(
                    thinking_budget=thinking_budget
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
                    "gemini",
                )

            return parsed_obj

        except Exception as exc:
            print(
                f"  ❌ Structured API call failed (attempt {attempt + 1}/{max_retries}): {exc}"
            )
            if attempt >= max_retries - 1:
                if _handle_api_error("gemini"):
                    return _query_gemini_structured(
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


def _query_openai_structured(
    prompt: str,
    system_message: str,
    response_schema: Type[T],
    model_cfg: DictConfig,
    use_cache: bool = True,
    max_retries: int = 3,
    temperature: Optional[float] = None,
) -> Optional[T]:
    """Call OpenAI with structured output using the Hydra model configuration."""
    client = app_config.get_client("openai")
    model_name = getattr(model_cfg, "name", app_config.model_config.openai.model_name)

    temp_eval = _cfg_attr(model_cfg, "temp_evaluation", app_config.model_config.temp_evaluation)
    generation_temperature = temperature if temperature is not None else temp_eval

    if use_cache:
        cached = _cache.get(prompt, system_message, model_name, generation_temperature, "openai")
        if cached:
            try:
                return response_schema.model_validate_json(cached)
            except Exception as exc:
                print(
                    f"  ⚠️ Cached response failed validation: {exc}. Requesting a fresh call."
                )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=generation_temperature,
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from OpenAI")

            parsed_obj = response_schema.model_validate_json(response_text)

            if use_cache:
                _cache.set(
                    prompt,
                    system_message,
                    model_name,
                    generation_temperature,
                    parsed_obj.model_dump_json(),
                    "openai",
                )

            return parsed_obj

        except Exception as exc:
            print(
                f"  ❌ Structured API call failed (attempt {attempt + 1}/{max_retries}): {exc}"
            )
            if attempt >= max_retries - 1:
                if _handle_api_error("openai"):
                    return _query_openai_structured(
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


def query_structured(
    prompt: str,
    system_message: str,
    response_schema: Type[T],
    model_cfg: DictConfig,
    use_cache: bool = True,
    max_retries: int = 3,
    temperature: Optional[float] = None,
) -> Optional[T]:
    """Provider-agnostic structured query dispatcher.
    
    Routes to the appropriate provider based on model_cfg.provider.
    """
    provider = getattr(model_cfg, "provider", app_config.model_config.provider)
    
    if provider == "gemini":
        return _query_gemini_structured(
            prompt, system_message, response_schema, model_cfg, use_cache, max_retries, temperature
        )
    elif provider == "openai":
        return _query_openai_structured(
            prompt, system_message, response_schema, model_cfg, use_cache, max_retries, temperature
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Expected 'gemini' or 'openai'.")


def _query_gemini_unstructured(
    prompt: str,
    system_message: str,
    model_cfg: DictConfig,
    temperature: float,
    use_cache: bool = True,
    max_retries: int = 3,
) -> str:
    """Call Gemini for free-form text using the Hydra model configuration."""
    client = app_config.get_client("gemini")
    model_name = getattr(model_cfg, "name", app_config.model_config.gemini.model_name)

    if use_cache:
        cached = _cache.get(prompt, system_message, model_name, temperature, "gemini")
        if cached:
            return cached

    for attempt in range(max_retries):
        try:
            generation_config = types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_message,
            )

            use_thinking = _cfg_attr(model_cfg, "use_thinking", app_config.model_config.use_thinking)
            if "2.5-pro" in model_name and use_thinking:
                thinking_budget = _cfg_attr(model_cfg, "thinking_budget", app_config.model_config.thinking_budget)
                generation_config.thinking_config = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )

            response = client.models.generate_content(
                model=f"models/{model_name}",
                contents=prompt,
                config=generation_config,
            )

            result_text = response.text

            if use_cache and result_text:
                _cache.set(prompt, system_message, model_name, temperature, result_text, "gemini")

            return result_text

        except Exception as exc:
            print(
                f"  ❌ Unstructured API call failed (attempt {attempt + 1}/{max_retries}): {exc}"
            )
            if attempt >= max_retries - 1:
                if _handle_api_error("gemini"):
                    return _query_gemini_unstructured(
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


def _query_openai_unstructured(
    prompt: str,
    system_message: str,
    model_cfg: DictConfig,
    temperature: float,
    use_cache: bool = True,
    max_retries: int = 3,
) -> str:
    """Call OpenAI for free-form text using the Hydra model configuration."""
    client = app_config.get_client("openai")
    model_name = getattr(model_cfg, "name", app_config.model_config.openai.model_name)

    if use_cache:
        cached = _cache.get(prompt, system_message, model_name, temperature, "openai")
        if cached:
            return cached

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )

            result_text = response.choices[0].message.content or ""

            if use_cache and result_text:
                _cache.set(prompt, system_message, model_name, temperature, result_text, "openai")

            return result_text

        except Exception as exc:
            print(
                f"  ❌ Unstructured API call failed (attempt {attempt + 1}/{max_retries}): {exc}"
            )
            if attempt >= max_retries - 1:
                if _handle_api_error("openai"):
                    return _query_openai_unstructured(
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


def query_text(
    prompt: str,
    system_message: str,
    model_cfg: DictConfig,
    temperature: float,
    use_cache: bool = True,
    max_retries: int = 3,
) -> str:
    """Provider-agnostic text query dispatcher.
    
    Routes to the appropriate provider based on model_cfg.provider.
    """
    provider = getattr(model_cfg, "provider", app_config.model_config.provider)
    
    if provider == "gemini":
        return _query_gemini_unstructured(
            prompt, system_message, model_cfg, temperature, use_cache, max_retries
        )
    elif provider == "openai":
        return _query_openai_unstructured(
            prompt, system_message, model_cfg, temperature, use_cache, max_retries
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Expected 'gemini' or 'openai'.")


# Backward compatibility aliases
query_gemini_structured = query_structured
query_gemini_unstructured = query_text


class EmbeddingClient:
    """Wrapper around the Gemini embeddings API with sync and async helpers.
    
    Note: Embeddings are Gemini-only. When used with other providers, 
    embedding generation will fall back to Gemini or gracefully no-op if unavailable.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        if not self.model_name.startswith("models/"):
            self.model_name = f"models/{self.model_name}"

    def get_embedding(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        if not text:
            return [] if isinstance(text, str) else [[] for _ in text]

        try:
            client = app_config.get_client("gemini")
        except Exception as exc:
            print(f"  ⚠️ Gemini client unavailable for embeddings: {exc}")
            is_single_item = isinstance(text, str)
            return [] if is_single_item else [[] for _ in text]

        is_single_item = isinstance(text, str)

        try:
            result = client.models.embed_content(model=self.model_name, contents=text)

            if is_single_item:
                return result.embeddings[0].values
            return [embedding.values for embedding in result.embeddings]

        except Exception as exc:
            print(f"  ❌ Failed to generate embedding: {exc}")
            if _handle_api_error("gemini"):
                return self.get_embedding(text)
            return [] if is_single_item else [[] for _ in text]

    async def get_embedding_async(
        self, text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_embedding, text)
