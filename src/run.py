"""Entry point that wires Hydra configuration and launches the evolution loop."""

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure the project root is on the import path when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import app_config
from src.evolution import ConceptEvolution
from src.webui.visualization import start_server


def read_file_content(filepath: str) -> str:
    """Read and return the contents of a UTF-8 text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: file not found: '{filepath}'")
        exit(1)
    except Exception as exc:
        print(f"❌ Error while reading file: {exc}")
        exit(1)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra-driven execution entry point for ConceptEvolve."""
    print("\n" + "=" * 70)
    print("🧬 CONCEPTEVOLVE – ALGORITHMIC CONCEPT EVOLUTION")
    print("=" * 70)
    print("\n📊 Experiment configuration:")
    print(OmegaConf.to_yaml(cfg))

    if getattr(cfg, "interactive_setup", False):
        print("\n--- Interactive Setup ---")
        default_provider = getattr(cfg.model, "provider", "gemini")
        provider_input = input(f"Select LLM provider [{default_provider}]: ").strip().lower()
        provider_choice = provider_input or default_provider
        OmegaConf.update(cfg, "model.provider", provider_choice, force_add=True)

        default_model_name = getattr(cfg.model, "name", "")
        model_input = input(f"Enter model name [{default_model_name}]: ").strip()
        model_choice = model_input or default_model_name
        if model_choice:
            OmegaConf.update(cfg, "model.name", model_choice, force_add=True)

        print("-------------------------\n")
        print("Updated configuration:")
        print(OmegaConf.to_yaml(cfg.model))

    # Configure the model from Hydra config and initialize the appropriate provider client.
    app_config.configure_model(cfg.model)
    provider = getattr(cfg.model, "provider", "gemini")
    print(f"🔧 Initializing {provider.upper()} client...")
    app_config.get_client(provider)

    problem_description = read_file_content(cfg.problem_file)
    if not problem_description:
        print(f"❌ Error: problem file '{cfg.problem_file}' is empty.")
        exit(1)

    print("\n🎯 Problem overview:")
    preview = problem_description[:200]
    ellipsis = "..." if len(problem_description) > 200 else ""
    print(f"   {preview}{ellipsis}")
    print("\n" + "=" * 70 + "\n")

    # Launch the realtime visualisation server in the Hydra working directory.
    search_root = os.getcwd()
    db_path_from_config = cfg.database.db_path
    db_abs_path = Path(search_root) / db_path_from_config

    port = 8000
    server_thread = threading.Thread(
        target=start_server,
        args=(port, str(db_abs_path)),
        daemon=True,
    )
    server_thread.start()
    time.sleep(1)

    url = f"http://localhost:{port}"
    print(f"🎨 Realtime evolution dashboard: {url}")
    try:
        webbrowser.open_new_tab(url)
    except Exception as exc:
        print(f"  (Unable to open the browser automatically: {exc})")

    evolution_process = ConceptEvolution(problem_description=problem_description, cfg=cfg)

    try:
        evolution_process.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Evolution interrupted by user")
        print("💾 Saving emergency checkpoint...")
        evolution_process.checkpoint_manager.save_checkpoint(
            evolution_process.db.get_all_programs(),
            evolution_process.current_generation,
            problem_description,
        )
        print("✅ Checkpoint saved. Resume with --resume")
    except Exception as exc:
        print(f"\n\n❌ Fatal error: {exc}")
        import traceback

        traceback.print_exc()
        print("\n💾 Attempting to save emergency checkpoint...")
        try:
            evolution_process.checkpoint_manager.save_checkpoint(
                evolution_process.db.get_all_programs(),
                evolution_process.current_generation,
                problem_description,
            )
            print("✅ Emergency checkpoint saved")
        except Exception:
            print("❌ Emergency checkpoint could not be saved")
        exit(1)


if __name__ == "__main__":
    main()
