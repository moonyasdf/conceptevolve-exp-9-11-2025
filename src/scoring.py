"""Adaptive scoring utilities for ConceptEvolve."""

from typing import List

import numpy as np

from src.concepts import ConceptScores


class AdaptiveScoring:
    """Maintain score history, normalise dimensions, and adapt weighting over time.

    Normalisation:
        * Each metric keeps a rolling history. Normalisation activates once at
          least five samples exist to avoid early-generation noise.
        * Scores are standardised with a z-score and rescaled into the [1, 10]
          range (centred at 5.5, scaled by 1.5). Results are clipped to remain
          compatible with agent prompts.

    Adaptive weighting:
        * Progress is estimated via ``generation / max_generations``.
        * ``w_novelty`` decays linearly from 0.40 → 0.30 while
          ``w_feasibility`` grows from 0.10 → 0.20. Potential and sophistication
          remain constant at 0.40 and 0.10 respectively before re-normalisation.
        * The combined score is multiplied by an alignment factor in [0.6, 1.0]
          derived from the alignment validator’s output.
    """

    def __init__(self, stagnation_threshold: int = 3, novelty_boost: float = 1.2) -> None:
        self.score_history = {
            "novelty": [],
            "potential": [],
            "sophistication": [],
            "feasibility": [],
        }
        self.best_score_history: List[float] = []
        self.stagnation_threshold = stagnation_threshold
        self.novelty_boost = novelty_boost

    def update_history(self, scores: ConceptScores) -> None:
        """Append the latest scores to the rolling history."""
        self.score_history["novelty"].append(scores.novelty)
        self.score_history["potential"].append(scores.potential)
        self.score_history["sophistication"].append(scores.sophistication)
        self.score_history["feasibility"].append(scores.feasibility)

    def normalize_scores(self, scores: ConceptScores) -> ConceptScores:
        """Normalise scores with a z-score when sufficient history exists."""
        if len(self.score_history["novelty"]) < 5:
            return scores

        normalized = {}
        for metric in ["novelty", "potential", "sophistication", "feasibility"]:
            values = self.score_history[metric]
            mean = float(np.mean(values))
            std = float(np.std(values)) + 1e-6
            raw_score = getattr(scores, metric)

            z_score = (raw_score - mean) / std
            scaled_score = 5.5 + (z_score * 1.5)
            normalized[metric] = max(1.0, min(10.0, scaled_score))

        return ConceptScores(**normalized)

    def update_best_score_history(self, best_score_this_gen: float) -> None:
        """Record the best combined score achieved in the current generation."""
        self.best_score_history.append(best_score_this_gen)

    def _is_stagnated(self) -> bool:
        """Detect whether top scores have plateaued for ``stagnation_threshold`` generations."""
        if len(self.best_score_history) < self.stagnation_threshold:
            return False

        last_scores = self.best_score_history[-self.stagnation_threshold :]
        if last_scores[-1] <= max(last_scores[:-1]):
            print("  📉 Stagnation detected! Increasing novelty weight...")
            return True
        return False

    def calculate_combined_score(
        self,
        scores: ConceptScores,
        generation: int,
        max_generations: int,
        alignment_score: float = 10.0,
    ) -> float:
        """Compute the generation-aware weighted score with alignment penalties."""
        progress = generation / max(max_generations, 1)

        w_novelty = 0.40 - (0.10 * progress)  # 0.40 -> 0.30
        w_potential = 0.40
        w_sophistication = 0.10
        w_feasibility = 0.10 + (0.10 * progress)  # 0.10 -> 0.20

        if self._is_stagnated():
            w_novelty *= self.novelty_boost

        total_weight = w_novelty + w_potential + w_sophistication + w_feasibility
        w_novelty /= total_weight
        w_potential /= total_weight
        w_sophistication /= total_weight
        w_feasibility /= total_weight

        base_score = (
            w_novelty * scores.novelty
            + w_potential * scores.potential
            + w_sophistication * scores.sophistication
            + w_feasibility * scores.feasibility
        )

        alignment_multiplier = 0.6 + (0.4 * (alignment_score / 10.0))
        final_score = base_score * alignment_multiplier

        return max(0.0, min(10.0, final_score))

    def get_score_statistics(self) -> dict:
        """Return descriptive statistics for each score dimension."""
        stats = {}
        for metric, values in self.score_history.items():
            if values:
                stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
            else:
                stats[metric] = None
        return stats
