# ConceptEvolve: An Architectural Deep Dive

This document provides a deeper, architectural overview of the ConceptEvolve framework. While the root `README.md` focuses on user-level operation, this guide is intended for developers, contributors, and those curious about the system's internal mechanics and design philosophy.

## Core Philosophy: Principled Exploration of the Design Space

At its heart, ConceptEvolve is not just an "idea generator." It is a systematic engine for exploring a complex **design space**. Traditional brainstorming is often random and prone to cognitive biases. This framework imposes a principled, evolutionary structure on the ideation process, turning creativity into a reproducible, stateful, and optimizable search.

The key principles are:
1.  **Stateful Evolution:** Every idea, critique, and score is persisted. This allows the system to build on past efforts, resume from interruptions, and enable complex, long-running explorations that would be impossible in a stateless context.
2.  **Structured Peer Review:** The Critic-Refiner loop is a simulation of academic or engineering peer review. By formalizing this adversarial process, we force ideas to become more robust, detailed, and defensible before they are considered "fit."
3.  **Controlled Speciation:** The Island Model is a deliberate strategy to combat premature convergence. It allows different "lineages" of a core idea to evolve in semi-isolation, fostering diversity and preventing a single, locally-optimal solution from dominating the entire population too early.

---

## Anatomy of an Evolutionary Cycle

To understand the architecture, it's best to follow the lifecycle of a single new concept within one generation:

1.  **Selection (The Genetic Source):** The cycle begins in `evolution.py` with a call to `self.db.sample()`. This delegates the selection logic to the database layer, which is crucial for decoupling.
    *   `src/database/parents.py`: A **parent** concept is chosen using a configurable strategy (e.g., `PowerLawSamplingStrategy`). This balances *exploitation* (favoring high-scoring concepts) and *exploration* (giving a chance to lower-ranked ones).
    *   `src/database/inspiration.py`: Several **inspiration** concepts are selected from two pools: a random sample from the "elite archive" and the deterministic top-k concepts. This provides both novel and high-quality genetic material for crossover.

2.  **Variation (The Creative Spark):** The selected concepts are passed to the `IdeaGenerator`.
    *   `src/prompts.py`: The `PromptSampler` randomly selects a mutation strategy (e.g., "Radical Reframe," "Synthesizer"). This injects dynamism into the creative process, preventing the LLM from falling into repetitive patterns.
    *   `src/agents.py`: `IdeaGenerator.mutate_or_crossover` synthesizes a new concept. If meta-learning is active, it also injects high-level recommendations into the prompt, guiding the search based on trends from previous generations.

3.  **Quality Control (The Crucible):** The raw concept is not immediately accepted. It enters a refinement loop.
    *   `ConceptCritic`: A skeptical agent identifies weaknesses, logical fallacies, and implementation bottlenecks.
    *   `IdeaGenerator.refine`: The generator is tasked with rewriting the concept to explicitly address the critiques. This iterative loop strengthens the idea significantly.

4.  **Novelty Filtering (The Diversity Gatekeeper):**
    *   `src/vector_index.py`: The new concept's embedding is checked against the FAISS index to quickly find its nearest semantic neighbors. This is an efficient, O(log n) pre-filter.
    *   `src/agents.py`: If a close neighbor is found, the `NoveltyJudge` agent performs a more nuanced, LLM-based comparison to determine if the new concept is a meaningful variation or a trivial duplicate. This prevents the population from being flooded with redundant ideas.

5.  **Fitness Evaluation (The Judgment):**
    *   `ConceptEvaluator`: Scores the concept on multiple axes (Novelty, Potential, etc.).
    *   `AlignmentValidator`: Ensures the concept is still a valid solution to the original problem statement, preventing "evolutionary drift."
    *   `src/scoring.py`: The `AdaptiveScoring` system normalizes the raw scores based on the history of all previous evaluations and computes a final `combined_score`. The weighting of scores can shift as the evolution progresses (e.g., favoring novelty early on and feasibility later).

6.  **Persistence and Indexing:** The validated, scored, and refined concept is saved to the SQLite database via `self.db.add()` and its embedding is added to the FAISS index.

7.  **Global Learning (The Long-Term Memory):**
    *   `src/meta_learning.py`: The `MetaSummarizer` consumes the newly evaluated concept. If enough concepts have accumulated, it runs a multi-step analysis to distill global insights and generate new, actionable recommendations that will influence the next generation's variation prompts.

---

## Key Architectural Pillars

-   **Hydra-Powered Configuration:** Centralizes all parameters, from LLM temperatures to database settings, allowing for easy experimentation and command-line overrides without code changes.
-   **Decoupled Database Logic:** The `ConceptDatabase` is more than a simple store; it's an active component of the evolutionary engine. By implementing selection, island management, and sampling logic within the database layer, the core `evolution.py` loop remains clean and focused on orchestration.
-   **Agentic Specialization:** Each agent in `src/agents.py` has a single, well-defined responsibility and a corresponding system prompt that primes it for its task (e.g., a skeptical reviewer, a creative researcher). This mirrors a high-functioning human team.
-   **Robust Structured Output:** The framework heavily relies on Gemini's native structured output capabilities, using Pydantic models (`src/concepts.py`) as the schema. This eliminates brittle regex parsing and ensures data integrity between agents.

## Extending the Framework

The system is designed to be modular. Here are common extension points:
-   **Add a New Mutation Strategy:** Simply add a new key-value pair to the `MUTATION_STRATEGIES` dictionary in `src/prompts.py`.
-   **Introduce a New Parent Selection Strategy:** Create a new class in `src/database/parents.py` that inherits from `ParentSamplingStrategy` and implement the `sample_parent` method. Then, add its name to the `parent_selection_strategy` choices in `src/database/dbase.py`.
-   **Integrate a Different LLM Provider:** The core logic is in `src/llm_utils.py`. One could create an abstract `LLMProvider` class and implement concrete versions (`GeminiProvider`, `AnthropicProvider`, etc.) to make the system multi-provider.