"""Gemini-powered agent pipeline for evolving algorithmic concepts.

This module orchestrates the end-to-end lifecycle of concept evolution: idea
creation, critique, refinement, scoring, novelty filtering, requirements
extraction, and alignment validation.  Each agent wraps a Gemini prompt strategy
and, where needed, a structured response schema that feeds downstream steps in
the evolution loop.
"""

# File: src/agents.py
# NOTE: Includes enhanced prompts, contextual refinement, and alignment validation.

from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from omegaconf import DictConfig
from src.concepts import (
    AlgorithmicConcept,
    ConceptScores,
    NoveltyDecision,
    SystemRequirements,
    VerificationReport,
)
from src.llm_utils import query_gemini_structured, query_gemini_unstructured
from src.prompts import PromptSampler
from src.prompts import (
    IDEA_GENERATION_EXAMPLES,
    SYSTEM_MSG_IDEA_GENERATOR,
    SYSTEM_MSG_CRITIC,
    SYSTEM_MSG_EVALUATOR,
    SYSTEM_MSG_REFINEMENT,
    SYSTEM_MSG_ALIGNMENT,
    SYSTEM_MSG_NOVELTY_JUDGE,
    SYSTEM_MSG_SOLVER,
    SYSTEM_MSG_VERIFIER,
)
from src.config import model_config

class IdeaGenerator:
    """Produce seed, mutated, and refined algorithmic concepts via Gemini prompts.

    The generator is responsible for the creative side of the pipeline.  It
    bootstraps first-generation ideas with few-shot exemplars, performs
    mutation/crossover against prior concepts, and rewrites descriptions after
    critiques.  Each stage relies on structured Gemini responses that can be
    promoted directly to :class:`AlgorithmicConcept` objects for downstream
    agents.
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
        long-form technical description.  The structured output is promoted to
        an :class:`AlgorithmicConcept`, providing the root node for downstream
        critique, scoring, and refinement agents.

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
        response = query_gemini_structured(
            prompt, 
            SYSTEM_MSG_IDEA_GENERATOR, 
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
        response = query_gemini_structured(
            prompt,
            SYSTEM_MSG_IDEA_GENERATOR,
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

    def refine(
        self, 
        concept: AlgorithmicConcept,
        all_critiques: List[str], 
        addressed_points: List[str], 
        problem_description: str,
        model_cfg: DictConfig  # Hydra model configuration for LLM calls
    ) -> Tuple[str, List[str]]:
        """Refine an existing concept using accumulated critiques and history.

        The refinement prompt threads together the current draft, every critique
        the concept has received, and a list of previously addressed issues.  By
        requesting a structured ``RefinementOutput`` schema, Gemini must produce
        both a fully rewritten description and a list of critique points tackled
        in the current iteration.  The caller merges ``newly_addressed_points``
        into ``addressed_points`` to drive multi-round refinement loops.

        Args:
            concept: The concept whose description is being rewritten.
            all_critiques: Chronological critiques sourced from
                :class:`ConceptCritic`.
            addressed_points: Critique snippets already resolved in prior loops.
            problem_description: Original user prompt for context anchoring.

        Returns:
            A tuple containing the refined description and an updated list of
            addressed critique points.
        """
        # Track which critiques are already resolved so Gemini can prioritize remaining gaps.
        addressed_summary = "\n- ".join(addressed_points) if addressed_points else "None yet"
        all_critiques_text = "\n\n---\n\n".join([
            f"**Critique Round {i+1}:**\n{c}"
            for i, c in enumerate(all_critiques)
        ])
        prompt = f"""
### 🎯 ORIGINAL PROBLEM
{problem_description}

### 📄 CURRENT DRAFT
**{concept.title}**
{concept.description}

### 🔍 CRITIQUE HISTORY
{all_critiques_text}

### ✅ POINTS ALREADY ADDRESSED
{addressed_summary}

### 📋 REFINEMENT TASK
1. Identify which critiques remain unresolved.
2. Rewrite the description to:
   - Resolve the outstanding critiques with concrete technical solutions.
   - Preserve and strengthen improvements already made.
   - Maintain the original creative insight.
   - Add any missing implementation detail.

3. Example of specificity:
   - ❌ "We will optimise runtime."
   - ✅ "Introduce an LRU cache with adaptive TTL based on query frequency, reducing lookups from O(n) to O(1)."

**IMPORTANT:** The refined description must replace the previous one entirely while keeping the innovative core.

**Output format:**
- refined_description: Fully rewritten description.
- newly_addressed_points: Bullet list of critiques resolved in this refinement.
"""

        class RefinementOutput(BaseModel):
            refined_description: str = Field(description="Complete refined description")
            newly_addressed_points: List[str] = Field(description="Critique points resolved in this refinement")
        
        # Pass the Hydra model configuration and the appropriate temperature.
        response = query_gemini_structured(
            prompt, 
            SYSTEM_MSG_REFINEMENT, 
            RefinementOutput,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_refinement
        )
        
        if response:
            new_addressed = addressed_points + response.newly_addressed_points
            return response.refined_description, new_addressed
        
        return concept.description, addressed_points

class ConceptCritic:
    """Deliver structured critiques that drive the refinement loop.

    The critic asks Gemini for an unstructured but highly prescriptive review of
    the concept, covering alignment, logic, feasibility, novelty, and potential
    failure modes.  Its textual feedback is fed to :meth:`IdeaGenerator.refine`
    so the generator can resolve outstanding issues in subsequent iterations.
    """
    
    def run(self, concept: AlgorithmicConcept, problem_description: str, model_cfg: DictConfig) -> str:
        """Request a detailed critique anchored to the original problem statement.

        The prompt enumerates five lenses—alignment, logical consistency,
        viability, novelty, and implementation risks—and instructs Gemini to
        produce bullet-point guidance.  Because the output is observational text
        rather than a structured schema, ``query_gemini_unstructured`` is used
        here.  The raw critique is appended to the concept's history for later
        refinements.

        Args:
            concept: Candidate concept being stress-tested.
            problem_description: Original problem description used to validate
                alignment.

        Returns:
            A markdown-compatible critique string.
        """
        prompt = f"""
### 🎯 ORIGINAL PROBLEM
{problem_description}

### 📄 PROPOSAL TO CRITIQUE
**{concept.title}**
{concept.description}

### 📋 TASK
Deliver a detailed critique that covers:

**1. Problem Alignment**
- Does it address every stated requirement?
- Does it drift away from the requested scope?

**2. Logical Soundness**
- Highlight reasoning gaps or unsupported steps.
- Flag contradictions or missing justifications.

**3. Technical Feasibility**
- Identify unrealistic assumptions or resource demands.
- Describe compute, data, or engineering bottlenecks.

**4. Genuine Novelty**
- Explain whether the idea is truly original or a superficial rebrand.
- Note when combinations of known techniques are insufficiently novel.

**5. Implementation Risks**
- Call out likely failure modes, edge cases, or hidden complexities.

**FORMAT:** Bullet list with precise technical reasoning.

**BE SPECIFIC:** Instead of "might have scalability issues", write "The streaming retriever requires O(n²) graph updates per query, which is infeasible beyond 10M documents".
"""
        
        # Unstructured call preserves free-form markdown critiques for human legibility.
        return query_gemini_unstructured(
            prompt, 
            SYSTEM_MSG_CRITIC,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_critique
        )

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
        return query_gemini_structured(
            prompt, 
            SYSTEM_MSG_EVALUATOR, 
            ConceptScores,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_evaluation
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
        return query_gemini_structured(
            prompt, 
            SYSTEM_MSG_NOVELTY_JUDGE, 
            NoveltyDecision,
            model_cfg=model_cfg,
            temperature=0.3
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
        return query_gemini_structured(
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
        result = query_gemini_structured(
            prompt,
            SYSTEM_MSG_ALIGNMENT,
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
    ) -> VerificationReport:
        """Return a structured verification verdict for the provided concept."""

        prompt = f"""
### 🎯 ORIGINAL PROBLEM
{problem_description}

### 📄 CONCEPT UNDER VERIFICATION
**{concept.title}**
{concept.description}

### 🕵️ TASK
Evaluate whether the concept is ready for execution. Provide a concise pass/fail
verdict, enumerate blocking issues that require correction, and suggest the most
impactful improvements.

Focus on feasibility, logical coherence, and alignment with the original
problem. If the concept passes, keep `blocking_issues` empty and use
`improvement_suggestions` for optional polish.
"""

        class VerificationPayload(BaseModel):
            passed: bool = Field(description="True when the concept is ready to proceed.")
            summary: str = Field(description="Short explanation of the decision.")
            blocking_issues: List[str] = Field(
                default_factory=list,
                description="Blocking problems preventing approval.",
            )
            improvement_suggestions: List[str] = Field(
                default_factory=list,
                description="Optional follow-up improvements.",
            )

        result = query_gemini_structured(
            prompt,
            SYSTEM_MSG_VERIFIER,
            VerificationPayload,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_critique,
        )

        if not result:
            return VerificationReport(
                passed=False,
                summary="Verifier did not return a structured response.",
                blocking_issues=["No verification response received."],
            )

        return VerificationReport(
            passed=bool(result.passed),
            summary=result.summary.strip(),
            blocking_issues=list(result.blocking_issues or []),
            improvement_suggestions=list(result.improvement_suggestions or []),
        )


class SolverAgent:
    """Agent responsible for correcting drafts after failed verification."""

    def correct(
        self,
        concept: AlgorithmicConcept,
        report: VerificationReport,
        problem_description: str,
        model_cfg: DictConfig,
    ) -> Tuple[Optional[str], List[str]]:
        """Produce an updated description that addresses verification blockers."""

        blocking_section = "\n".join(f"- {item}" for item in report.blocking_issues) or "- None"
        improvements_section = "\n".join(f"- {item}" for item in report.improvement_suggestions) or "- None"
        prompt = f"""
### 🎯 ORIGINAL PROBLEM
{problem_description}

### 📄 CURRENT DRAFT
**{concept.title}**
{concept.description}

### ❌ VERIFICATION RESULT
Summary: {report.summary or "No summary provided."}

Blocking issues:
{blocking_section}

Suggested improvements:
{improvements_section}

### 🛠️ CORRECTION TASK
Write a new, fully-specified description that resolves every blocking issue while
preserving the concept's core insight. Provide a bullet changelog explaining the
fixes you applied.

Return structured JSON with:
- corrected_description: the rewritten concept
- applied_fixes: bullet list describing the key changes
"""

        class CorrectionOutput(BaseModel):
            corrected_description: str = Field(description="Rewritten concept description.")
            applied_fixes: List[str] = Field(
                default_factory=list,
                description="Bullet list describing the modifications made.",
            )

        result = query_gemini_structured(
            prompt,
            SYSTEM_MSG_SOLVER,
            CorrectionOutput,
            model_cfg=model_cfg,
            temperature=model_cfg.temp_refinement,
        )

        if not result or not result.corrected_description:
            return None, []

        return result.corrected_description, list(result.applied_fixes or [])
