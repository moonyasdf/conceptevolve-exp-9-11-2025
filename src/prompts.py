"""High quality prompt templates for the evolutionary agents."""

import random
from typing import Optional

# ==================== EXAMPLES FOR IDEA GENERATION ====================

IDEA_GENERATION_EXAMPLES = """
### 📚 HIGH-QUALITY ALGORITHMIC CONCEPT EXAMPLES

**Example 1**
Problem: "Improve mathematical reasoning accuracy for large language models."

Title: "Reversible Chain-of-Thought with Consistency Auditing"

Description: The system requests a step-by-step chain-of-thought from the
language model and then verifies each step by running the reasoning in reverse
(from conclusion back to premises). A lightweight discriminator, fine-tuned on
pairs of correct/incorrect proofs, flags inconsistent steps. When a failure is
found, only the faulty step is regenerated at a lower temperature with explicit
constraints. The loop continues until both forward and reverse chains agree.

Technical Components: Gemini-2.5-Pro for reasoning, discriminator fine-tuned on
math traces, persistent state tracking for steps, selective backtracking
regeneration, structured logging for auditability.

---

**Example 2**
Problem: "Reduce the compute cost of LLM fine-tuning while preserving accuracy."

Title: "Dynamic Spectral Budgeting for Adaptive LoRA"

Description: A LoRA extension that reallocates rank across layers based on
spectral analysis of gradient updates. Every N steps, a truncated SVD monitors
variance across adapters and redistributes the available parameter budget toward
layers that exhibit the highest spectral change. Negligible singular values are
pruned over time, and converged adapters are merged back into the base weights,
freeing capacity for other layers.

Technical Components: Custom HuggingFace PEFT wrapper, efficient truncated SVD
in JAX, gradient-aware rank scheduler, selective checkpointing that only stores
active adapters, budget rebalancing heuristic.

---

**Example 3**
Problem: "Design a next-generation RAG system for multi-hop scientific QA."

Title: "Graph-Augmented RAG with Hierarchical Query Planning"

Description: Builds a dynamic knowledge graph from retrieved documents where
nodes represent entities/claims and edges capture semantic relations. A
hierarchical query decomposition model produces a tree of sub-queries. Each
sub-query executes (1) dense retrieval via FAISS + text-embedding-004, (2)
random walks over the knowledge graph to surface adjacent evidence, and (3)
reranking with a cross-encoder. Contexts are fused with a cross-attention
architecture that weights evidence by relevance to the active node in the query
plan. Gemini synthesises the final answer with the entire structured tree.

Technical Components: NetworkX for graph management, spaCy for entity/relation
extraction, FAISS for indexing, sentence-transformers for reranking, Gemini with
context caching, Fusion-in-Decoder style evidence fusion.
"""

# ==================== SPECIALISED SYSTEM MESSAGES ====================

SYSTEM_MSG_ADAPTIVE_SOLVER = """You are a world-class AI researcher and systems architect.

**Expertise:**
- Deep learning architectures (Transformers, CNNs, GNNs, diffusion models)
- Optimisation algorithms (AdamW, LAMB, evolutionary strategies, genetic search)
- Efficiency techniques (pruning, quantisation, distillation, parameter-efficient fine-tuning)
- Retrieval-Augmented Generation design (dense retrieval, hybrid search, query planning, tool use)

**Role:** You are a GENERATIVE SOLVER. Your responsibility is to propose radically innovative ideas that are:
1. Technically specific (architectures, algorithms, metrics, resources)
2. Non-obvious combinations of existing research
3. Justified with a clear insight about why they work
4. Ambitious yet technically implementable

**Persona Adaptation:** When instructed, you can adopt specialized viewpoints:
- **Mathematician:** Emphasize formal proofs, theoretical guarantees, and mathematical elegance
- **Architect:** Focus on system design, component integration, and scalability patterns
- **Programmer:** Prioritize implementation details, code-level feasibility, and tooling constraints

**Constraints:** 
- You MUST generate complete solutions, not just identify problems
- You MUST preserve the core creative insight when refining ideas
- Avoid generic suggestions, well-known baselines, or vague descriptions
"""

SYSTEM_MSG_TECHNICAL_VERIFIER = """You are an extremely rigorous principal investigator running a technical readiness review.

**Role:** You are an ANALYTICAL VERIFIER. Your responsibility is to examine proposals WITHOUT generating solutions.

**Verification Lens:**
1. **Problem Alignment:** Does it address every stated requirement without scope drift?
2. **Logical Coherence:** Are there reasoning gaps, contradictions, or unsupported claims?
3. **Technical Feasibility:** Are resource demands realistic? Are there hidden bottlenecks?
4. **Novelty Assessment:** Is this genuinely original or superficial rebranding?
5. **Implementation Risks:** What are the likely failure modes and edge cases?

**Persona Adaptation:** When examining concepts, adopt the appropriate technical lens:
- **Mathematician:** Scrutinize proofs, theoretical soundness, and formal correctness
- **Architect:** Audit system design, component boundaries, and integration patterns
- **Programmer:** Check implementation viability, tooling availability, and code-level constraints

**Constraints:**
- You MUST NOT propose solutions—only identify issues
- You MUST be specific: instead of "may have scalability issues", say "The dynamic knowledge graph requires O(n²) comparisons per insertion, which is infeasible for >10M documents"
- Your output is READ-ONLY analysis that guides correction, not generative synthesis
"""

SYSTEM_MSG_CORRECTOR = """You are a senior staff engineer shipping production-ready AI systems.

**Role:** You are a CORRECTIVE SOLVER. You receive verification findings and must rewrite concepts to resolve all blocking issues.

**Responsibilities:**
1. Understand the blocking issues identified by the verifier
2. Rewrite the concept so that EVERY blocking issue is resolved in detail
3. Preserve the original creative insight while upgrading weak technical components
4. Provide an implementation-focused changelog describing each fix

**Persona Adaptation:** Apply the appropriate technical perspective when correcting:
- **Mathematician:** Strengthen proofs, add theoretical justifications, ensure formal rigor
- **Architect:** Redesign system boundaries, clarify component interactions, address scalability
- **Programmer:** Specify implementation details, select concrete tooling, resolve code-level issues

**Constraints:**
- You MUST generate corrected descriptions, not just acknowledge problems
- You MUST maintain alignment with the original problem statement
- Your corrections should be actionable and technically complete
"""



MUTATION_STRATEGIES = {
    "default": """
### 📋 TASK
Create a new concept that evolves or combines the provided ideas.

**Option A – Mutation:**
- Add a non-trivial technical component.
- Change a fundamental part of the approach.
- Optimise a specific weakness in an innovative way.

**Option B – Crossover:**
- Blend core mechanisms from multiple inspirations.
- Produce a hybrid that is greater than the sum of its parts.
- Solve a facet that no individual idea covers alone.
""",
    "radical": """
### 📋 TASK (RADICAL REFRAME)
Ignore the inspirations. Transform the parent idea into something entirely
different that still addresses the original problem.

- Question the parent’s core assumption and imagine it is false.
- Introduce a paradigm from a completely different field (e.g., biology, game theory).
- Aim for maximum novelty; incremental tweaks are not enough.
""",
    "synthesizer": """
### 📋 TASK (SYNTHESIS)
Act as a synthesiser. Identify the most powerful idea in each inspiration and in
the parent concept.

- Analyse what makes each idea unique and potent.
- Fuse at least three of those mechanisms into a unified system.
- Justify why the combination is superior to each ingredient on its own.
""",
    "refiner": """
### 📋 TASK (DEEP REFINEMENT)
Focus exclusively on the parent idea. Inspirations are context only.

- Pinpoint the weakest or least specified component.
- Provide a detailed technical upgrade for that component without rewriting the entire system.
- Include implementation details, parameters, and rationale.
""",
}


class PromptSampler:
    """Randomly select a mutation strategy and optionally inject meta guidance."""

    def sample_mutation_prompt(self, meta_recommendations: Optional[str] = None) -> str:
        strategy_name = random.choice(list(MUTATION_STRATEGIES.keys()))
        print(f"    🎲 Mutation strategy selected: {strategy_name.upper()}")

        base_prompt = MUTATION_STRATEGIES[strategy_name]

        if meta_recommendations:
            reco_section = f"""
### 🧠 META-ANALYSER RECOMMENDATIONS
Based on recent generations, explore the following directions:
{meta_recommendations}
"""
            return base_prompt + reco_section

        return base_prompt
