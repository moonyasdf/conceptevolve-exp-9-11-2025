import getpass
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from google import genai
from openai import OpenAI


@dataclass
class GeminiSettings:
    model_name: str = "gemini-2.5-pro"
    embedding_model: str = "text-embedding-004"
    use_thinking: bool = True
    thinking_budget: int = -1
    response_mime_type: str = "application/json"
    enable_json_schema: bool = True


@dataclass
class OpenAISettings:
    model_name: str = "grok-code-fast-1"
    base_url: Optional[str] = None
    response_format: str = "json_object"
    enable_json_schema: bool = True


class ModelConfig:
    """Configuration for LLM model behavior supporting multiple providers."""

    def __init__(self) -> None:
        self.provider: str = os.getenv("MODEL_PROVIDER", "gemini").lower()

        self.temperatures: Dict[str, float] = {
            "generation": float(os.getenv("MODEL_TEMP_GEN", "1.0")),
            "evaluation": float(os.getenv("MODEL_TEMP_EVAL", "0.3")),
            "critique": float(os.getenv("MODEL_TEMP_CRIT", "0.5")),
            "refinement": float(os.getenv("MODEL_TEMP_REF", "0.7")),
        }
        self.temp_generation = self.temperatures["generation"]
        self.temp_evaluation = self.temperatures["evaluation"]
        self.temp_critique = self.temperatures["critique"]
        self.temp_refinement = self.temperatures["refinement"]

        self.top_p = float(os.getenv("MODEL_TOP_P", "0.95"))
        self.top_k = int(os.getenv("MODEL_TOP_K", "40"))

        self.gemini = GeminiSettings(
            model_name=os.getenv("GEMINI_MODEL", os.getenv("MODEL_NAME", "gemini-2.5-pro")),
            embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
            use_thinking=os.getenv("GEMINI_USE_THINKING", "true").lower() == "true",
            thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "-1")),
            response_mime_type=os.getenv("GEMINI_RESPONSE_MIME_TYPE", "application/json"),
            enable_json_schema=os.getenv("GEMINI_ENABLE_JSON_SCHEMA", "true").lower() == "true",
        )
        self.openai = OpenAISettings(
            model_name=os.getenv("OPENAI_MODEL", os.getenv("MODEL_NAME", "grok-code-fast-1")),
            base_url=os.getenv("OPENAI_BASE_URL"),
            response_format=os.getenv("OPENAI_RESPONSE_FORMAT", "json_object"),
            enable_json_schema=os.getenv("OPENAI_ENABLE_JSON_SCHEMA", "true").lower() == "true",
        )

    def _get_attr(self, cfg: Any, key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def update_from_config(self, cfg: Any) -> None:
        if cfg is None:
            return

        provider = self._get_attr(cfg, "provider", self.provider)
        if provider:
            self.provider = str(provider).lower()

        name = self._get_attr(cfg, "name", None)
        if name:
            if self.provider == "gemini":
                self.gemini.model_name = name
            elif self.provider == "openai":
                self.openai.model_name = name

        gemini_model = self._get_attr(cfg, "gemini_model", None)
        if gemini_model:
            self.gemini.model_name = gemini_model

        openai_model = self._get_attr(cfg, "openai_model", None)
        if openai_model:
            self.openai.model_name = openai_model

        base_url = self._get_attr(cfg, "base_url", None)
        if base_url:
            self.openai.base_url = base_url

        embedding_model = self._get_attr(cfg, "embedding_model", None)
        if embedding_model:
            self.gemini.embedding_model = embedding_model

        use_thinking = self._get_attr(cfg, "use_thinking", None)
        if use_thinking is not None:
            self.gemini.use_thinking = bool(use_thinking)

        thinking_budget = self._get_attr(cfg, "thinking_budget", None)
        if thinking_budget is not None:
            self.gemini.thinking_budget = int(thinking_budget)

        response_mime_type = self._get_attr(cfg, "response_mime_type", None)
        if response_mime_type:
            self.gemini.response_mime_type = response_mime_type

        use_json_schema = self._get_attr(cfg, "use_json_schema", None)
        if use_json_schema is not None:
            flag = bool(use_json_schema)
            self.gemini.enable_json_schema = flag
            self.openai.enable_json_schema = flag

        openai_response_format = self._get_attr(cfg, "openai_response_format", None)
        if openai_response_format:
            self.openai.response_format = openai_response_format

        for attr_name, key in [
            ("temp_generation", "temp_generation"),
            ("temp_evaluation", "temp_evaluation"),
            ("temp_critique", "temp_critique"),
            ("temp_refinement", "temp_refinement"),
        ]:
            value = self._get_attr(cfg, key, None)
            if value is not None:
                numeric = float(value)
                setattr(self, attr_name, numeric)
                mapping_key = (
                    "generation"
                    if attr_name == "temp_generation"
                    else attr_name.replace("temp_", "")
                )
                self.temperatures[mapping_key] = numeric

    @property
    def model_name(self) -> str:
        return self.openai.model_name if self.provider == "openai" else self.gemini.model_name

    @property
    def embedding_model(self) -> str:
        return self.gemini.embedding_model

    @property
    def response_mime_type(self) -> str:
        return self.gemini.response_mime_type

    @property
    def use_json_schema(self) -> bool:
        return (
            self.openai.enable_json_schema if self.provider == "openai" else self.gemini.enable_json_schema
        )

    @property
    def openai_response_format(self) -> str:
        return self.openai.response_format

    @property
    def openai_base_url(self) -> Optional[str]:
        return self.openai.base_url

    @property
    def use_thinking(self) -> bool:
        return self.gemini.use_thinking

    @property
    def thinking_budget(self) -> int:
        return self.gemini.thinking_budget


class AppConfig:
    """Singleton responsible for API credentials and client management across providers."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.gemini_api_key = None
            cls._instance.openai_api_key = None
            cls._instance.gemini_client = None
            cls._instance.openai_client = None
            cls._instance.model_config = ModelConfig()
        return cls._instance

    def configure_model(self, cfg: Any) -> None:
        self.model_config.update_from_config(cfg)

    def set_gemini_api_key(self, key: str) -> None:
        """Configure the Gemini client with the provided API key."""
        self.gemini_api_key = key
        try:
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            print("✅ Google API key configured successfully.")
        except Exception as exc:
            print(f"❌ Failed to configure the Google API key: {exc}")
            self.gemini_api_key = None
            self.gemini_client = None

    def set_openai_api_key(self, key: str, base_url: Optional[str] = None) -> None:
        """Configure the OpenAI client with the provided API key and optional base URL."""
        self.openai_api_key = key
        if base_url:
            self.model_config.openai.base_url = base_url
        try:
            if base_url:
                self.openai_client = OpenAI(api_key=self.openai_api_key, base_url=base_url)
            else:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            print("✅ OpenAI API key configured successfully.")
        except Exception as exc:
            print(f"❌ Failed to configure the OpenAI API key: {exc}")
            self.openai_api_key = None
            self.openai_client = None

    def get_client(self, provider: str = "gemini"):
        """Return a cached client for the specified provider, prompting for credentials if required."""
        provider = (provider or "gemini").lower()
        if provider == "gemini":
            return self._get_gemini_client()
        if provider == "openai":
            return self._get_openai_client()
        raise ValueError(f"Unknown provider: {provider}. Expected 'gemini' or 'openai'.")

    def _get_gemini_client(self):
        if self.gemini_client:
            return self.gemini_client

        key_from_env = os.getenv("GOOGLE_API_KEY")
        if key_from_env:
            print("🔑 Found Google API key in the GOOGLE_API_KEY environment variable.")
            self.set_gemini_api_key(key_from_env)
            return self.gemini_client

        print("🔑 Google API key not found.")
        while not self.gemini_client:
            try:
                key_input = getpass.getpass("Please enter your Google API key and press Enter: ")
                if key_input:
                    self.set_gemini_api_key(key_input)
                else:
                    print("The API key cannot be empty.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled. Exiting.")
                exit()
        return self.gemini_client

    def _get_openai_client(self):
        if self.openai_client:
            return self.openai_client

        key_from_env = os.getenv("OPENAI_API_KEY")
        base_url = self.model_config.openai_base_url or os.getenv("OPENAI_BASE_URL")

        if key_from_env:
            print("🔑 Found OpenAI API key in the OPENAI_API_KEY environment variable.")
            self.set_openai_api_key(key_from_env, base_url)
            return self.openai_client

        print("🔑 OpenAI API key not found.")
        while not self.openai_client:
            try:
                key_input = getpass.getpass("Please enter your OpenAI API key and press Enter: ")
                if key_input:
                    if not base_url:
                        base_url = input("Enter OpenAI base URL (press Enter to skip): ").strip() or None
                    self.set_openai_api_key(key_input, base_url)
                else:
                    print("The API key cannot be empty.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled. Exiting.")
                exit()
        return self.openai_client


# Global singleton instances
app_config = AppConfig()
model_config = app_config.model_config
