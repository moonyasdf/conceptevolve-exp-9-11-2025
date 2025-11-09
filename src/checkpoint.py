import json
import os
from typing import List, Tuple

from src.concepts import AlgorithmicConcept


class CheckpointManager:
    """Persist and restore the evolutionary state between runs."""

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        population: List[AlgorithmicConcept],
        generation: int,
        problem_description: str = "",
    ) -> None:
        """Serialize the current population and generation index to disk."""
        checkpoint_file = os.path.join(self.checkpoint_dir, f"gen_{generation:03d}.json")

        data = {
            "generation": generation,
            "problem_description": problem_description,
            "population_size": len(population),
            "population": [c.model_dump() for c in population],
        }

        try:
            with open(checkpoint_file, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
            print(f"  💾 Saved checkpoint: gen_{generation:03d}.json")
        except Exception as exc:
            print(f"  ⚠️ Failed to save checkpoint: {exc}")

    def load_latest_checkpoint(self) -> Tuple[List[AlgorithmicConcept], int, str]:
        """Load the most recent checkpoint if one exists."""
        checkpoints = [
            filename
            for filename in os.listdir(self.checkpoint_dir)
            if filename.startswith("gen_") and filename.endswith(".json")
        ]

        if not checkpoints:
            return [], 0, ""

        latest = sorted(checkpoints)[-1]
        checkpoint_file = os.path.join(self.checkpoint_dir, latest)

        try:
            with open(checkpoint_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)

            population = [AlgorithmicConcept(**concept) for concept in data.get("population", [])]
            generation = data.get("generation", 0)
            problem_description = data.get("problem_description", "")

            print(f"  📂 Loaded checkpoint: {latest}")
            print(f"     Generation: {generation}")
            print(f"     Population size: {len(population)} concepts")

            return population, generation, problem_description

        except Exception as exc:
            print(f"  ❌ Failed to load checkpoint: {exc}")
            return [], 0, ""

    def list_checkpoints(self) -> List[str]:
        """Return a sorted list of checkpoint filenames."""
        checkpoints = [
            filename
            for filename in os.listdir(self.checkpoint_dir)
            if filename.startswith("gen_") and filename.endswith(".json")
        ]
        return sorted(checkpoints)

    def delete_old_checkpoints(self, keep_last_n: int = 5) -> None:
        """Delete older checkpoints, keeping only the most recent ``keep_last_n``."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_last_n:
            return

        to_delete = checkpoints[:-keep_last_n]

        for checkpoint in to_delete:
            try:
                os.remove(os.path.join(self.checkpoint_dir, checkpoint))
                print(f"  🗑️ Deleted old checkpoint: {checkpoint}")
            except Exception as exc:
                print(f"  ⚠️ Failed to delete {checkpoint}: {exc}")
