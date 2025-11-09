# NOTE: The cache implementation lives in ``src/llm_utils.py``.
# This module is kept for documentation and backwards compatibility with earlier imports.

"""
The caching system is implemented in ``src/llm_utils.py`` to avoid circular imports and
centralize all LLM integrations.

Capabilities:
- Hash-based cache keyed by (model, temperature, prompt, system message).
- JSON files persisted under the ``.cache/`` directory.
- Toggleable via the ``use_cache`` flag on each query function.
- Supports both structured and unstructured Gemini responses.
"""
