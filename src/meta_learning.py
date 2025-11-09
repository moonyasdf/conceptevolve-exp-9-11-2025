"""Meta-learning loop that distils insights from recent generations."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig

from src.concepts import AlgorithmicConcept
from src.llm_utils import query_text
from src.prompts_meta import (
    META_STEP1_SYSTEM_MSG,
    META_STEP1_USER_MSG,
    META_STEP2_SYSTEM_MSG,
    META_STEP2_USER_MSG,
    META_STEP3_SYSTEM_MSG,
    META_STEP3_USER_MSG,
)

logger = logging.getLogger(__name__)


def construct_concept_summary_for_meta(concept: AlgorithmicConcept) -> str:
    """Build a concise summary that feeds the meta-learning prompts."""
    if concept is None:
        return "None."

    score_str = f"Combined Score: {concept.combined_score:.2f}"
    if concept.scores:
        scores = concept.scores
        score_str += (
            f" (N: {scores.novelty:.1f}, P: {scores.potential:.1f}, "
            f"S: {scores.sophistication:.1f}, F: {scores.feasibility:.1f})"
        )

    if concept.critique_history:
        latest_critique = concept.critique_history[-1]
        critique_summary = " ".join(latest_critique.split()[:50]) + "..."
    else:
        critique_summary = "N/A"

    return (
        f"**Title:** {concept.title}\n"
        f"**Description:** {concept.description[:400]}...\n"
        f"**Performance:** {score_str}\n"
        f"**Final Critique Summary:** {critique_summary}"
    )


class MetaSummarizer:
    """Manage multi-step meta-analysis to guide future mutations."""

    def __init__(self, model_cfg: DictConfig, max_recommendations: int = 5):
        self.model_cfg = model_cfg
        self.max_recommendations = max_recommendations
        self.meta_summary: Optional[str] = None
        self.meta_scratch_pad: Optional[str] = None
        self.meta_recommendations: Optional[str] = None
        self.evaluated_since_last_meta: List[AlgorithmicConcept] = []
        self.total_concepts_processed = 0

    def add_evaluated_concept(self, concept: AlgorithmicConcept) -> None:
        """Queue a newly evaluated concept for the next meta-analysis cycle."""
        self.evaluated_since_last_meta.append(concept)

    def should_update_meta(self, meta_interval: Optional[int]) -> bool:
        """Return ``True`` when enough concepts have accumulated for meta-analysis."""
        if meta_interval is None or meta_interval <= 0:
            return False
        return len(self.evaluated_since_last_meta) >= meta_interval

    def update_meta_memory(
        self, best_concept: Optional[AlgorithmicConcept] = None
    ) -> Tuple[Optional[str], float]:
        """Run the three-step meta-learning pipeline."""
        if not self.evaluated_since_last_meta:
            return None, 0.0

        print(
            f"  🧠 Running meta-learning cycle with {len(self.evaluated_since_last_meta)} new concepts..."
        )

        individual_summaries, cost1 = self._step1_individual_summaries()
        if not individual_summaries:
            logger.error("Meta-learning Step 1 failed: no summaries produced.")
            return None, 0.0

        global_insights, cost2 = self._step2_global_insights(individual_summaries, best_concept)
        if not global_insights:
            logger.error("Meta-learning Step 2 failed: no global insights produced.")
            return None, cost1

        recommendations, cost3 = self._step3_generate_recommendations(global_insights, best_concept)
        if not recommendations:
            logger.error("Meta-learning Step 3 failed: no recommendations produced.")
            return None, cost1 + cost2

        self.meta_summary = (
            individual_summaries
            if self.meta_summary is None
            else f"{self.meta_summary}\n\n{individual_summaries}"
        )
        self.meta_scratch_pad = global_insights
        self.meta_recommendations = recommendations

        processed = len(self.evaluated_since_last_meta)
        self.total_concepts_processed += processed
        self.evaluated_since_last_meta = []

        total_cost = cost1 + cost2 + cost3
        print(f"  ✅ Meta-learning cycle complete. Estimated cost: ${total_cost:.4f}")
        return self.meta_recommendations, total_cost

    def _step1_individual_summaries(self) -> Tuple[Optional[str], float]:
        """Summarise each concept individually."""
        summaries: List[str] = []
        total_cost = 0.0
        for concept in self.evaluated_since_last_meta:
            prompt = META_STEP1_USER_MSG.format(
                concept_summary=construct_concept_summary_for_meta(concept)
            )
            summary = query_text(
                prompt,
                META_STEP1_SYSTEM_MSG,
                self.model_cfg,
                self.model_cfg.temp_evaluation,
            )
            if summary:
                summaries.append(summary)
        return "\n\n".join(summaries), total_cost

    def _step2_global_insights(
        self, summaries: str, best_concept: Optional[AlgorithmicConcept]
    ) -> Tuple[Optional[str], float]:
        """Aggregate global insights from individual summaries."""
        prompt = META_STEP2_USER_MSG.format(
            individual_summaries=summaries,
            previous_insights=self.meta_scratch_pad or "None.",
            best_concept_info=construct_concept_summary_for_meta(best_concept) if best_concept else "None.",
        )
        insights = query_text(
            prompt,
            META_STEP2_SYSTEM_MSG,
            self.model_cfg,
            self.model_cfg.temp_evaluation,
        )
        return insights, 0.0

    def _step3_generate_recommendations(
        self, insights: str, best_concept: Optional[AlgorithmicConcept]
    ) -> Tuple[Optional[str], float]:
        """Transform insights into concrete recommendations."""
        prompt = META_STEP3_USER_MSG.format(
            global_insights=insights,
            previous_recommendations=self.meta_recommendations or "None.",
            max_recommendations=self.max_recommendations,
            best_concept_info=construct_concept_summary_for_meta(best_concept) if best_concept else "None.",
        )
        recommendations = query_text(
            prompt,
            META_STEP3_SYSTEM_MSG,
            self.model_cfg,
            self.model_cfg.temp_evaluation,
        )
        return recommendations, 0.0

    def get_current_recommendations(self) -> Optional[str]:
        """Expose the latest recommendation string for prompt injection."""
        return self.meta_recommendations

    def save_state(self, filepath: str) -> None:
        """Persist meta-learning state to disk."""
        state = {
            "meta_summary": self.meta_summary,
            "meta_scratch_pad": self.meta_scratch_pad,
            "meta_recommendations": self.meta_recommendations,
            "total_concepts_processed": self.total_concepts_processed,
            "evaluated_since_last_meta": [
                concept.model_dump() for concept in self.evaluated_since_last_meta
            ],
        }
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
        except Exception as exc:
            logger.error(f"Failed to save meta-learning state: {exc}")

    def load_state(self, filepath: str) -> None:
        """Restore meta-learning state if a previous run exists."""
        if not Path(filepath).exists():
            return
        try:
            with open(filepath, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            self.meta_summary = state.get("meta_summary")
            self.meta_scratch_pad = state.get("meta_scratch_pad")
            self.meta_recommendations = state.get("meta_recommendations")
            self.total_concepts_processed = state.get("total_concepts_processed", 0)
            self.evaluated_since_last_meta = [
                AlgorithmicConcept(**concept)
                for concept in state.get("evaluated_since_last_meta", [])
            ]
            print("  🧠 Meta-learning state restored.")
        except Exception as exc:
            logger.error(f"Failed to load meta-learning state: {exc}")
