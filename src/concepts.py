from typing import List, Optional
import uuid

from pydantic import BaseModel, Field


# --- Structured output schemas for Gemini ---


class ConceptScores(BaseModel):
    """Fitness scores returned by the evaluator agent."""

    novelty: float = Field(
        default=5.0,
        description="Score from 1 to 10 indicating how original and unconventional the idea is.",
    )
    potential: float = Field(
        default=5.0,
        description="Score from 1 to 10 reflecting the expected uplift over current best practices.",
    )
    sophistication: float = Field(
        default=5.0,
        description="Score from 1 to 10 capturing the technical depth and elegance of the concept.",
    )
    feasibility: float = Field(
        default=5.0,
        description="Score from 1 to 10 estimating how practical the idea is with current technology.",
    )


class NoveltyDecision(BaseModel):
    """Binary decision used by the novelty judge."""

    is_novel: bool = Field(
        default=True,
        description="True when the concept is materially different, False when it is a trivial variant.",
    )
    explanation: str = Field(
        default="",
        description="Concise justification describing why the concept was marked novel or duplicate.",
    )


class SystemRequirements(BaseModel):
    """Engineering requirements extracted from a concept description."""

    core_components: List[str] = Field(
        default_factory=list,
        description="Concrete hardware/software components required for execution (e.g., 'NVIDIA A100 24GB').",
    )
    discovered_sub_problems: List[str] = Field(
        default_factory=list,
        description="Follow-up research tasks or challenges identified while analyzing the concept.",
    )


class VerificationReport(BaseModel):
    """Structured outcome emitted by the verifier agent for each round."""

    round_index: int = Field(default=0, description="1-based index of the verification round.")
    passed: bool = Field(default=False, description="True when the concept passes verification.")
    issue_summaries: List[str] = Field(
        default_factory=list,
        description="Concise descriptions of the issues or gaps identified during verification.",
    )
    diagnostics: Optional[str] = Field(
        default=None,
        description="Optional diagnostic notes providing evidence, persona-specific context, or traceability.",
    )


# --- Primary data structure stored in the population ---


class AlgorithmicConcept(BaseModel):
    """Represents an individual concept in the evolutionary population."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(default="")
    description: str = Field(default="")

    # Revision history
    draft_history: List[str] = Field(default_factory=list)
    verification_reports: List[VerificationReport] = Field(default_factory=list)

    # Extracted outputs
    system_requirements: SystemRequirements = Field(default_factory=SystemRequirements)

    # Evolution metadata
    generation: int = Field(default=0)
    parent_id: Optional[str] = Field(default=None)
    inspiration_ids: List[str] = Field(default_factory=list)
    island_idx: Optional[int] = Field(default=None)
    embedding: List[float] = Field(default_factory=list)

    # Fitness scores
    scores: Optional[ConceptScores] = Field(default=None)
    combined_score: float = Field(default=0.0)

    def get_full_prompt_text(self) -> str:
        """Return a canonical representation used when generating embeddings."""
        return f"## Title: {self.title}\n\n### Description\n{self.description}"
