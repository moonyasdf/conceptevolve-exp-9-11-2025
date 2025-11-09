"""Gemini-powered agent pipeline for evolving algorithmic concepts.

This module orchestrates the end-to-end lifecycle of concept evolution: idea
creation, verification, scoring, novelty filtering, requirements extraction, and
alignment validation. Each agent wraps a Gemini prompt strategy and, where
needed, a structured response schema that feeds downstream steps in the evolution
loop.
"""

# File: src/agents.py
# NOTE: Includes enhanced prompts, contextual correction, and alignment validation.

import contextlib
from pydantic import BaseModel, Field
from typing import List, Optional
from omegaconf import DictConfig
from src.concepts import (
    AlgorithmicConcept,
    ConceptScores,
    NoveltyDecision,
    SystemRequirements,
    VerificationReport,
)
from src.llm_utils import query_structured
from src.prompts import PromptSampler
from src.prompts import (
    IDEA_GENERATION_EXAMPLES,
    SYSTEM_MSG_ADAPTIVE_SOLVER,
    SYSTEM_MSG_TECHNICAL_VERIFIER,
    SYSTEM_MSG_CORRECTOR,
)

EVALUATOR_SYSTEM_PROMPT = """You are a program committee member scoring concepts
from 1.0 to 10.0 on four axes:

**Novelty (40%)** – From "established idea" to "never-before-seen insight".
**Potential (40%)** – From "unlikely to beat baselines" to "could redefine the field".
**Sophistication (10%)** – From "trivial" to "technically intricate and well integrated".
**Feasibility (10%)** – From "not currently buildable" to "achievable with standard resources".

Prioritise bold creativity even if feasibility is moderate.
"""

ALIGNMENT_SYSTEM_PROMPT = """You are an impartial alignment examiner verifying
that the proposal truly solves the original problem.

Score each axis from 0.0 to 10.0:
1. coverage_score – How completely does it address every explicit requirement?
2. scope_fidelity_score – Does it stay within scope without drifting?
3. directness_score – Is it a direct solution versus an indirect assistive tool?

Compute alignment_score = mean of the three scores and set is_aligned = true
when alignment_score ≥ 6.0. Provide a brief, specific explanation.
"""

NOVELTY_SYSTEM_PROMPT = """You are an expert in NLP semantic similarity.
Determine whether two algorithm descriptions represent the same idea.

**Consider the same idea (is_novel = false) when:**
- Core algorithm is identical despite wording differences.
- Main components are the same with only superficial tweaks.
- One description is merely more detailed than the other.

**Consider them different (is_novel = true) when:**
- Architectures or mechanisms are fundamentally distinct.
- Additional non-trivial components change the approach.
- The combination of techniques creates a new workflow.

Return a structured response with `is_novel` and a concise explanation.
"""


class SolverAgent:
    """Produce seed, mutated, and corrected algorithmic concepts via Gemini prompts.

    The solver agent is responsible for the generative side of the pipeline. It
    bootstraps first-generation ideas with few-shot exemplars, performs
    mutation/crossover against prior concepts, and corrects concepts based on
    verification feedback. Each stage relies on structured Gemini responses that
    can be promoted directly to :class:`AlgorithmicConcept` objects for
    downstream agents.
    """

    def __init__(self):
        self.prompt_sampler = PromptSampler()

    def generate_initial_concept(
        self, 
        problem_description: str, 
        generation: int,
        model_cfg: DictConfig,  # Hydra model configuration for LLM calls
        use_cache: bool = True
    ) -> Optional[AlgorithmicConcept]:
        """Generate a seed concept from the problem description via few-shot prompting.

        The prompt concatenates ``IDEA_GENERATION_EXAMPLES`` with the user brief
        and requests a structured ``InitialIdea`` payload containing a title and
        long-form technical description. The structured output is promoted to an
        :class:`AlgorithmicConcept`, providing the root node for downstream
        verification, scoring, and selection agents.

        Args:
            problem_description: Natural-language description of the task the
                concept must solve.
            generation: Index used to tag the resulting concept within the
                evolutionary run.

        Returns:
            An :class:`AlgorithmicConcept` if Gemini produces a complete
            ``InitialIdea`` payload, otherwise ``None`` when generation fails.
        """
        prompt = f"""
{IDEA_GENERATION_EXAMPLES}

### 🎯 YOUR TURN

**User Problem:**
{problem_description}

**Instructions:**
Propose an algorithmic concept that:
1. Is technically specific (name architectures, algorithms, metrics, resources).
2. Blends at least two existing ideas in a non-obvious way.
3. Highlights a key insight explaining why the approach could succeed.
4. Is ambitious yet implementable with current technology.
5. Includes concrete technical components throughout the description.

**Required output:**
- title: A precise, technically descriptive title for the concept.
- description: A detailed explanation (minimum 200 words) covering:
  * The primary mechanism or workflow.
  * Specific technical components and how they interact.
  * The key insight that justifies performance gains.
  * Implementation considerations (data, compute, architecture choices).
"""

        class InitialIdea(BaseModel):
            title: str = Field(description="Technically descriptive name for the concept")
            description: str = Field(description="Detailed technical description (>=200 words)")
        
        # Pass the Hydra model configuration and appropriate temperature to the helper.
        response = query_structured(
            prompt, 
            SYSTEM_MSG_ADAPTIVE_SOLVER, 
            InitialIdea,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_generation,
            use_cache=use_cache
        )
        
        if response and response.title and response.description:
            return AlgorithmicConcept(
                title=response.title,
                description=response.description,
                generation=generation
            )
        return None

    def mutate_or_crossover(
        self,
        parent: AlgorithmicConcept,
        inspirations: List[AlgorithmicConcept],
        generation: int,
        problem_description: str,
        model_cfg: DictConfig,  # Hydra model configuration for LLM calls
        meta_recommendations: Optional[str] = None,
    ) -> Optional[AlgorithmicConcept]:
        """Mutate or crossover a parent concept with inspiration concepts.

        A dynamic mutation task sampled from :class:`PromptSampler` adds
        variability to the crossover instructions.  The prompt composes the
        parent, a set of inspirational snippets, and explicit novelty
        guardrails before requesting a structured ``MutatedIdea`` response.  The
        structured schema keeps downstream agents agnostic to whether the concept
        originated from mutation or fresh generation.

        Args:
            parent: Concept selected as the primary genome to evolve.
            inspirations: Additional concepts that supply contrasting ideas or
                building blocks for crossover.
            generation: Target generation index for the offspring.
            problem_description: Original user problem used to anchor the
                mutation instructions.
            meta_recommendations: Optional meta-learning insights to inject into
                the mutation prompt.

        Returns:
            A novel :class:`AlgorithmicConcept` instance or ``None`` if Gemini
            fails to produce a differentiated idea.
        """
        inspiration_texts = "\n\n".join([
            f"**Inspiration {i+1}:** {insp.title}\n{insp.description[:300]}..."
            for i, insp in enumerate(inspirations)
        ])
        mutation_task = self.prompt_sampler.sample_mutation_prompt(
            meta_recommendations=meta_recommendations
        )
        prompt = f"""
### 🎯 ORIGINAL PROBLEM
{problem_description}

### 👨‍👩‍👧 PARENT CONCEPT (TO EVOLVE)
**{parent.title}**
{parent.description}

### 💡 INSPIRATION CONCEPTS
{inspiration_texts}

{mutation_task}

**REQUIREMENTS:**
1. The new concept must be significantly different from the parent (not just more detail).
2. Introduce at least one entirely new technical component.
3. Maintain alignment with the original problem statement.
4. Provide a technically specific description.

**Output format:**
- title: Reflect the novelty of the new concept.
- description: Detailed technical specification of the new approach.
"""
        
        class MutatedIdea(BaseModel):
            title: str
            description: str

        # Structured response ensures mutated concepts have the same interface as fresh seeds.
        response = query_structured(
            prompt,
            SYSTEM_MSG_ADAPTIVE_SOLVER,
            MutatedIdea,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_generation
        )

        if response and response.title and response.description:
            return AlgorithmicConcept(
                title=response.title,
                description=response.description,
                generation=generation,
                parent_id=parent.id,
                inspiration_ids=[insp.id for insp in inspirations]
            )
        return None

    def correct(
        self,
        concept: AlgorithmicConcept,
        report: VerificationReport,
        problem_description: str,
        model_cfg: DictConfig,
        persona: Optional[str] = None,
    ) -> Optional[AlgorithmicConcept]:
        """Produce an updated concept that addresses verification blockers.
        
        This method consumes verification feedback from the VerifierAgent and
        produces a revised concept description that resolves all blocking issues
        while preserving the core creative insight. Unlike the legacy refine()
        pathway, correct() operates on structured VerificationReport data with
        explicit issue summaries and diagnostics.
        
        Args:
            concept: The concept whose description needs correction.
            report: Structured VerificationReport containing issue summaries and diagnostics.
            problem_description: Original problem statement for context anchoring.
            model_cfg: Hydra model configuration for LLM calls.
            persona: Optional override to force a specific correction persona.
            
        Returns:
            An updated :class:`AlgorithmicConcept` with the applied corrections, or ``None`` when
            no revision could be produced.
        """
        issues_section = "\n".join(f"- {item}" for item in report.issue_summaries) or "- None"
        diagnostics_section = report.diagnostics or "No additional diagnostics provided."

        persona_pref = persona
        if persona_pref is None and model_cfg is not None:
            try:
                persona_pref = getattr(model_cfg, "solver_persona")
            except AttributeError:
                persona_pref = None
            if not persona_pref:
                with contextlib.suppress(Exception):
                    persona_pref = model_cfg.get("solver_persona")  # type: ignore[attr-defined]

        persona_block = ""
        if persona_pref:
            persona_block = f"\n### 🎭 PERSONA\nAdopt the vantage point of a {persona_pref.strip()}. Use this perspective to prioritise the corrections.\n"
        
        prompt = f"""
### 🎯 ORIGINAL PROBLEM
{problem_description}

### 📄 CURRENT DRAFT
**{concept.title}**
{concept.description}

### ❌ VERIFICATION RESULT
Passed: {report.passed}
Round: {report.round_index}

Issues identified:
{issues_section}

Diagnostics:
{diagnostics_section}
{persona_block}
### 🛠️ CORRECTION TASK
Write a new, fully-specified concept that resolves every identified issue while
preserving the concept's core insight. Provide a bullet changelog explaining the
fixes you applied.

Return structured JSON with:
- title: the updated, technically precise concept title
- description: the fully revised concept description
- change_log: bullet list summarising the key changes
"""

        class CorrectionOutput(BaseModel):
            title: str = Field(description="Updated concept title after correction.")
            description: str = Field(description="Rewritten concept description.")
            change_log: List[str] = Field(
                default_factory=list,
                description="Bullet list describing the modifications made.",
            )

        result = query_structured(
            prompt,
            SYSTEM_MSG_CORRECTOR,
            CorrectionOutput,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_refinement,
        )

        if not result or not result.description:
            return None

        if result.title and result.title.strip():
            concept.title = result.title

        if result.description and result.description.strip():
            if concept.description != result.description:
                concept.description = result.description
                if (
                    not concept.draft_history
                    or concept.draft_history[-1] != result.description
                ):
                    concept.draft_history.append(result.description)

        change_log = list(result.change_log or [])
        if change_log:
            if concept.verification_reports:
                latest_report = concept.verification_reports[-1]
                applied = "\n".join(f"- {entry}" for entry in change_log)
                if latest_report.diagnostics:
                    latest_report.diagnostics = (
                        f"{latest_report.diagnostics}\n\nApplied fixes:\n{applied}"
                    )
                else:
                    latest_report.diagnostics = f"Applied fixes:\n{applied}"
            print("  🔧 Correction change log:")
            for item in change_log:
                print(f"    - {item}")

        return concept


class ConceptEvaluator:
    """Score concepts across novelty, potential impact, sophistication, and viability.

    The evaluator converts Gemini's structured assessment into a
    :class:`ConceptScores` Pydantic model.  These scores feed selection logic in
    the evolution core and influence which concepts continue to future
    generations.
    """
    
    def run(self, concept: AlgorithmicConcept, model_cfg: DictConfig) -> Optional[ConceptScores]:
        """Ask Gemini to quantify multiple fitness dimensions for a concept.

        The prompt outlines concrete rubrics for novelty, potential impact,
        sophistication, and viability.  ``query_gemini_structured`` enforces the
        :class:`ConceptScores` schema so downstream selection operators receive
        normalized floats instead of free-form prose.

        Args:
            concept: Concept under evaluation.

        Returns:
            A :class:`ConceptScores` instance or ``None`` if Gemini cannot
            respect the schema.
        """
        prompt = f"""
### 📄 IDEA TO EVALUATE
**{concept.title}**
{concept.description}

### 📊 SCORING RUBRIC (1.0 to 10.0)
Score the idea on each dimension below. Prioritise bold creativity while still
rewarding grounded execution plans.

**Novelty (40%)**
- 9-10: Breakthrough insight or unprecedented synthesis.
- 7-8: Non-obvious combination or significant twist on prior work.
- 5-6: Interesting extension of known techniques.
- 3-4: Minor variation on established methods.
- 1-2: Well-known idea with no differentiation.

**Potential (40%)**
- 9-10: Could redefine the field if successful.
- 7-8: Likely to beat the current state of the art by a clear margin (>5%).
- 5-6: Offers modest improvements (2-5%).
- 3-4: Benefits are uncertain or marginal.
- 1-2: Unlikely to outperform existing baselines.

**Sophistication (10%)**
- 9-10: Deep, multi-component technical synthesis.
- 7-8: Solid engineering with several non-trivial parts.
- 5-6: Technically correct but straightforward.
- 3-4: Simple or shallow implementation.
- 1-2: Superficial description lacking detail.

**Feasibility (10%)**
- 9-10: Achievable with standard resources (single-digit GPUs, weeks).
- 7-8: Reasonable resource lift (4-8 GPUs or similar infrastructure).
- 5-6: Significant investment but still plausible.
- 3-4: Requires exceptional resources or tooling.
- 1-2: Not realistically buildable today.

**IMPORTANT:** Return a JSON object using the exact keys `novelty`, `potential`,
`sophistication`, and `feasibility`, each containing a float in [1.0, 10.0].
"""
        
        # Structured evaluation ensures each rubric is returned as a normalized float.
        return query_structured(
            prompt,
            EVALUATOR_SYSTEM_PROMPT,
            ConceptScores,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_evaluation,
        )

class NoveltyJudge:
    """Detect semantic duplicates so the population maintains diversity.

    The judge compares two candidate concepts and returns a structured
    :class:`NoveltyDecision` describing whether they are meaningfully distinct.
    This signal helps the population manager avoid promoting near-identical
    concepts.
    """
    
    def run(
        self, 
        concept1: AlgorithmicConcept, 
        concept2: AlgorithmicConcept,
        model_cfg: DictConfig
    ) -> Optional[NoveltyDecision]:
        """Request a binary novelty decision between two concepts.

        Gemini receives both titles and descriptions plus explicit decision
        criteria and must respond with ``is_novel`` and an explanation via the
        :class:`NoveltyDecision` schema.  The low temperature favors consistent
        duplicate detection.

        Args:
            concept1: First concept to compare.
            concept2: Second concept to compare.

        Returns:
            A structured novelty verdict or ``None`` when the schema is not met.
        """
        
        prompt = f"""
### 🔍 CONCEPT COMPARISON

**Concept 1**
Title: {concept1.title}
Description: {concept1.description}

**Concept 2**
Title: {concept2.title}
Description: {concept2.description}

### ❓ QUESTION
Do these two concepts represent the same fundamental idea?

**Treat them as the SAME idea (`is_novel = false`) when:**
- The core algorithm is identical despite different terminology.
- Main components are the same with only superficial tweaks.
- One description is simply more detailed than the other.

**Treat them as DIFFERENT ideas (`is_novel = true`) when:**
- Architectures or mechanisms differ in a meaningful way.
- Additional non-trivial components change how the system works.
- The combination of techniques leads to a distinct workflow.

**Output format:**
- `is_novel`: boolean flag indicating whether the concepts are meaningfully different.
- `explanation`: 2-3 sentences justifying the decision.
"""
        
        # Structured decision keeps novelty filtering machine-readable for the FAISS index manager.
        return query_structured(
            prompt,
            NOVELTY_SYSTEM_PROMPT,
            NoveltyDecision,
            model_cfg=model_cfg,
            temperature=0.3,
        )

class RequirementsExtractor:
    """Map concepts to concrete system requirements and research sub-problems.

    The extractor converts descriptive concepts into actionable engineering
    checklists and open research questions.  The resulting
    :class:`SystemRequirements` objects help the execution planner scope future
    work.
    """
    
    def run(self, concept: AlgorithmicConcept, model_cfg: DictConfig) -> Optional[SystemRequirements]:
        """Derive implementation components and follow-up research tasks.

        Gemini receives the concept description along with examples of the level
        of specificity expected for ``core_components`` and
        ``discovered_sub_problems``.  ``query_gemini_structured`` guarantees that
        the response fits the :class:`SystemRequirements` schema.

        Args:
            concept: Concept whose requirements are being extracted.

        Returns:
            A :class:`SystemRequirements` instance or ``None`` when Gemini cannot
            honor the schema.
        """
        
        prompt = f"""
### 📄 ALGORITHM DESCRIPTION
**{concept.title}**
{concept.description}

### 📋 TASK
Extract:

**1. core_components** – Concrete software/hardware assets required to build the system.
   Examples:
   - "NVIDIA A100 (24GB+) or equivalent accelerator"
   - "FAISS vector database with IVF-PQ index"
   - "text-embedding-004 model from Google Gemini"
   - "HuggingFace Transformers >= 4.30 for fine-tuning adapters"
   - "ms-marco-MiniLM cross-encoder for reranking"

**2. discovered_sub_problems** – New research or engineering challenges surfaced by the concept.
   Examples:
   - "Design a context fusion metric that balances relevance and diversity"
   - "Scale dynamic knowledge graphs to >10M nodes without performance collapse"
   - "Collect supervision data for a discourse-coherence discriminator"
   - "Develop a selective backtracking planner that avoids regenerating solved steps"

Be specific and technical. Avoid generic phrases like "needs a GPU"; prefer
"GPU with Ampere architecture and at least 24GB VRAM".
"""
        
        # Structured extraction returns normalized component lists for scheduling and build planning.
        return query_structured(
            prompt, 
            "You are an AI systems architect specialising in extracting technical requirements.", 
            SystemRequirements,
            model_cfg=model_cfg,
            temperature=0.3
        )


class AlignmentValidator:
    """Check problem-solution alignment before promoting a concept.

    The validator scores whether a concept directly addresses the original user
    problem, using Gemini's structured output to return per-dimension scores plus
    a boolean verdict.  This gate keeps the pipeline focused on relevant ideas
    despite aggressive mutation.
    """
    
    def run(
        self, 
        concept: AlgorithmicConcept, 
        problem_description: str,
        model_cfg: DictConfig
    ) -> Tuple[bool, str, float]:
        """Evaluate coverage, scope fidelity, and directness for a concept.

        The prompt instructs Gemini to reason about alignment across three axes,
        compute an average, and emit a structured payload with scores, verdict,
        and rationale.  The tuple returned here is consumed by the evolution
        loop to drop misaligned ideas early.

        Args:
            concept: Concept under validation.
            problem_description: Original problem statement to validate against.

        Returns:
            Tuple of ``(is_aligned, explanation, alignment_score)`` where
            ``alignment_score`` is the mean of the three axis scores.
        """
        prompt = f"""
### 🎯 ORIGINAL PROBLEM
{problem_description}

### 💡 PROPOSED SOLUTION
**{concept.title}**
{concept.description}

### 📋 EVALUATION TASK
Determine whether the proposed solution directly addresses the problem.
Score each dimension from 0.0 to 10.0.

1. **coverage_score** – How completely are all explicit requirements covered?
   - 10.0: Fully covers every requirement
   - 5.0: Partially covers the requirements
   - 0.0: Ignores key requirements

2. **scope_fidelity_score** – Does the solution stay within scope?
   - 10.0: Completely focused on the stated problem
   - 5.0: Addresses the problem but with significant scope drift
   - 0.0: Solves a different problem altogether

3. **directness_score** – Is the solution direct versus tangential?
   - 10.0: Direct solution
   - 5.0: Requires substantial extra steps
   - 0.0: Only assists someone else in solving the problem

**Computation:**
- alignment_score = (coverage_score + scope_fidelity_score + directness_score) / 3
- is_aligned = true when alignment_score ≥ 6.0; otherwise false

Return JSON with keys:
- coverage_score: float
- scope_fidelity_score: float
- directness_score: float
- alignment_score: float
- is_aligned: boolean
- explanation: 2-3 sentences with concrete justification
"""
        
        class AlignmentResult(BaseModel):
            coverage_score: float
            scope_fidelity_score: float
            directness_score: float
            alignment_score: float
            is_aligned: bool
            explanation: str

        # Use a low, fixed temperature for deterministic alignment scoring.
        result = query_structured(
            prompt,
            ALIGNMENT_SYSTEM_PROMPT,
            AlignmentResult,
            model_cfg=model_cfg,
            temperature=0.3,
        )

        if result:
            return result.is_aligned, result.explanation, result.alignment_score

        return True, "Validation error - accepted by default", 6.0


class VerifierAgent:
    """Gatekeeper that determines whether a concept is ready to progress."""

    def verify(
        self,
        concept: AlgorithmicConcept,
        problem_description: str,
        model_cfg: DictConfig,
        persona: Optional[str] = None,
    ) -> VerificationReport:
        """Return a structured verification verdict for the provided concept.
        
        This method performs analytical verification WITHOUT generating solutions.
        It examines feasibility, logical coherence, and alignment with the original
        problem statement, then returns a structured VerificationReport with
        issue summaries and optional diagnostics.
        
        Args:
            concept: The concept under verification.
            problem_description: Original problem statement for context.
            model_cfg: Hydra model configuration for LLM calls.
            persona: Optional override to force a specific verifier persona (mathematician, architect, programmer).
            
        Returns:
            A VerificationReport containing the pass/fail verdict, issue summaries, and diagnostics.
        """
        
        persona_pref = persona
        if persona_pref is None and model_cfg is not None:
            try:
                persona_pref = getattr(model_cfg, "verifier_persona")
            except AttributeError:
                persona_pref = None
            if not persona_pref:
                with contextlib.suppress(Exception):
                    persona_pref = model_cfg.get("verifier_persona")  # type: ignore[attr-defined]

        persona_block = ""
        if persona_pref:
            persona_block = f"\n### 🎭 PERSONA\nExamine this concept from the vantage point of a {persona_pref.strip()}. Use this perspective when identifying issues and forming your verdict.\n"

        prompt = f"""
### 🎯 ORIGINAL PROBLEM
{problem_description}

### 📄 CONCEPT UNDER VERIFICATION
**{concept.title}**
{concept.description}
{persona_block}
### 🕵️ TASK
Evaluate whether the concept is ready for execution. Provide a concise pass/fail
verdict, enumerate issues that require correction, and provide diagnostic evidence.

Focus on feasibility, logical coherence, and alignment with the original
problem. You MUST NOT propose solutions - only identify gaps and provide diagnostic notes.

Return structured JSON with:
- passed: boolean indicating whether the concept is ready to proceed
- issue_summaries: list of concise issue descriptions (empty if passing)
- diagnostics: optional string providing evidence, context, or traceability notes
"""

        class VerificationPayload(BaseModel):
            passed: bool = Field(description="True when the concept is ready to proceed.")
            issue_summaries: List[str] = Field(
                default_factory=list,
                description="Concise descriptions of issues or gaps (empty if passing).",
            )
            diagnostics: Optional[str] = Field(
                default=None,
                description="Optional diagnostic notes providing evidence or persona-specific context.",
            )

        result = query_structured(
            prompt,
            SYSTEM_MSG_TECHNICAL_VERIFIER,
            VerificationPayload,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_critique,
        )

        if not result:
            return VerificationReport(
                passed=False,
                issue_summaries=["No verification response received."],
                diagnostics="The verifier did not return a structured response.",
            )

        return VerificationReport(
            passed=bool(result.passed),
            issue_summaries=list(result.issue_summaries or []),
            diagnostics=result.diagnostics,
        )
