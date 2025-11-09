import getpass
import os

from google import genai


class ModelConfig:
    """Default configuration for Gemini model behavior."""

    def __init__(self) -> None:
        # Primary generative model (only gemini-2.5-pro or gemini-2.0-flash-exp are supported)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

        # Task-specific temperatures
        self.temp_generation = float(os.getenv("GEMINI_TEMP_GEN", "1.0"))
        self.temp_evaluation = float(os.getenv("GEMINI_TEMP_EVAL", "0.3"))
        self.temp_critique = float(os.getenv("GEMINI_TEMP_CRIT", "0.5"))
        self.temp_refinement = float(os.getenv("GEMINI_TEMP_REF", "0.7"))

        # General sampling parameters
        self.top_p = float(os.getenv("GEMINI_TOP_P", "0.95"))
        self.top_k = int(os.getenv("GEMINI_TOP_K", "40"))

        # Thinking configuration (only respected by gemini-2.5-pro family)
        self.use_thinking = os.getenv("GEMINI_USE_THINKING", "true").lower() == "true"
        # -1 = dynamic, 0 = disabled, 128-32768 = fixed budget
        self.thinking_budget = int(os.getenv("GEMINI_THINKING_BUDGET", "-1"))

        # Embedding model used across the pipeline
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")


class AppConfig:
    """Singleton responsible for API credentials and Gemini client reuse."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.api_key = None
            cls._instance.client = None
            cls._instance.model_config = ModelConfig()
        return cls._instance

    def set_api_key(self, key: str) -> None:
        """Configure the Gemini client with the provided API key."""
        self.api_key = key
        try:
            self.client = genai.Client(api_key=self.api_key)
            print("✅ Google API key configured successfully.")
        except Exception as exc:
            print(f"❌ Failed to configure the API key: {exc}")
            self.api_key = None
            self.client = None

    def get_client(self):
        """Return a cached Gemini client, prompting for credentials if required."""
        if self.client:
            return self.client

        key_from_env = os.getenv("GOOGLE_API_KEY")
        if key_from_env:
            print("🔑 Found Google API key in the GOOGLE_API_KEY environment variable.")
            self.set_api_key(key_from_env)
            return self.client

        print("🔑 Google API key not found.")
        while not self.client:
            try:
                key_input = getpass.getpass("Please enter your Google API key and press Enter: ")
                if key_input:
                    self.set_api_key(key_input)
                else:
                    print("The API key cannot be empty.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled. Exiting.")
                exit()
        return self.client


# Global singleton instances
app_config = AppConfig()
model_config = app_config.model_config
