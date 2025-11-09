# src/evolution.py

# NOTE: Refactored to use the new database architecture, island model, Hydra,
# and to propagate model configuration to every agent.

import os
import math
import logging
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Tuple, Union

from tqdm import tqdm
from omegaconf import DictConfig

from src.database.dbase import ConceptDatabase, DatabaseConfig
from src.agents import (
    IdeaGenerator,
    ConceptCritic,
    ConceptEvaluator,
    NoveltyJudge,
    RequirementsExtractor,
    AlignmentValidator,
    SolverAgent,
    VerifierAgent,
)
from src.concepts import AlgorithmicConcept, ConceptScores, VerificationReport
from src.llm_utils import EmbeddingClient
from src.vector_index import ConceptIndex
from src.scoring import AdaptiveScoring
from src.checkpoint import CheckpointManager
from src.meta_learning import MetaSummarizer

logger = logging.getLogger(__name__)


class GenerationSkipped(Exception):
    """Raised when a generation is intentionally skipped without fatal error."""


class ConceptEvolution:
    def __init__(self, problem_description: str, cfg: DictConfig):
        self.problem_description = problem_description
        self.cfg = cfg
        self.current_generation = 0

        self.idea_generator = IdeaGenerator()
        self.critic = ConceptCritic()
        self.evaluator = ConceptEvaluator()
        self.novelty_judge = NoveltyJudge()
        self.req_extractor = RequirementsExtractor()
        self.alignment_validator = AlignmentValidator()
        self.verifier = VerifierAgent()
        self.solver = SolverAgent()

        # MEJORA: Pasamos el nombre del modelo de embedding desde la config de Hydra
        self.embedding_client = EmbeddingClient(model_name=cfg.model.embedding_model)

        self.scoring_system = AdaptiveScoring()

        self.db_config = self._build_database_config(cfg)
        self.db = ConceptDatabase(self.db_config)

        self.embedding_dimension = self._determine_embedding_dimension()
        try:
            self.concept_index = ConceptIndex(dimension=self.embedding_dimension)
        except Exception as exc:
            logger.warning(
                "Unable to initialize ConceptIndex with dimension %s: %s. Falling back to 768.",
                self.embedding_dimension,
                exc,
            )
            self.embedding_dimension = 768
            self.concept_index = ConceptIndex(dimension=self.embedding_dimension)

        self.checkpoint_manager = CheckpointManager(cfg.checkpoint_dir)

        self.meta_interval = getattr(self.cfg.evolution, "meta_interval", 0) or 0
        self.meta_state_path = Path(cfg.checkpoint_dir).resolve() / "meta_state.json"
        self.meta_summarizer: Optional[MetaSummarizer] = None
        self.current_meta_recommendations: Optional[str] = None
        if self.meta_interval > 0:
            self.meta_summarizer = MetaSummarizer(model_cfg=cfg.model)
            try:
                self.meta_summarizer.load_state(str(self.meta_state_path))
                self.current_meta_recommendations = self.meta_summarizer.get_current_recommendations()
            except Exception as exc:
                logger.warning(
                    "Failed to load meta-learning state from %s: %s",
                    self.meta_state_path,
                    exc,
                )

        configured_workers = getattr(self.cfg.evolution, "max_parallel_workers", 1)
        try:
            self.max_workers = max(1, int(configured_workers))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid value for evolution.max_parallel_workers (%s). Falling back to 1 worker.",
                configured_workers,
            )
            self.max_workers = 1

        self.jobs_submitted = 0
        self.jobs_completed = 0
        self._last_successful_generation = self.db.get_max_generation()

        self._index_lock = threading.RLock()
        self._meta_lock = threading.RLock()
        self._scoring_lock = threading.RLock()

    def _build_database_config(self, cfg: DictConfig) -> DatabaseConfig:
        if not hasattr(cfg, "database") or cfg.database is None:
            raise ValueError(
                "Hydra configuration must include a 'database' section with the required parameters."
            )

        try:
            db_config = DatabaseConfig.from_omegaconf(cfg.database)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Database configuration failed validation: {exc}") from exc

        db_path_value = Path(db_config.db_path)
        if not db_path_value.is_absolute():
            db_path_value = Path(os.getcwd()) / db_path_value

        try:
            db_config.apply_overrides(db_path=str(db_path_value))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Database configuration failed validation: {exc}") from exc

        db_config.validate()
        return db_config

    def _get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if not text:
            return []
        # MEJORA: `get_embedding` ahora puede manejar lotes.
        return self.embedding_client.get_embedding(text)

    def _infer_dimension_from_concepts(self, concepts: List[AlgorithmicConcept]) -> Optional[int]:
        for concept in concepts or []:
            if concept and concept.embedding:
                length = len(concept.embedding)
                if length:
                    return length
        return None

    def _determine_embedding_dimension(self) -> int:
        configured = (
            getattr(self.cfg.model, "embedding_dimension", None)
            if hasattr(self.cfg, "model")
            else None
        )
        if configured is not None:
            try:
                configured_value = int(configured)
                if configured_value > 0:
                    return configured_value
                logger.warning(
                    "Configured embedding dimension must be positive (received %s).",
                    configured,
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid configured embedding dimension: %s",
                    configured,
                )

        existing_concepts = self._safe_get_all_concepts()
        inferred = self._infer_dimension_from_concepts(existing_concepts)
        if inferred:
            return inferred

        logger.warning(
            "Unable to determine embedding dimension. Falling back to 768. Set `model.embedding_dimension` in your config.",
        )
        return 768

    def _rebuild_concept_index(
        self, existing_concepts: Optional[List[AlgorithmicConcept]] = None
    ) -> None:
        concepts = existing_concepts if existing_concepts is not None else self._safe_get_all_concepts()
        with self._index_lock:
            inferred = self._infer_dimension_from_concepts(concepts)
            if inferred and inferred != self.embedding_dimension:
                logger.info(
                    "Updating embedding index dimension from %s to %s based on existing data.",
                    self.embedding_dimension,
                    inferred,
                )
                self.embedding_dimension = inferred
            try:
                self.concept_index = ConceptIndex(dimension=self.embedding_dimension)
            except Exception as exc:
                logger.warning(
                    "Failed to reinstate ConceptIndex with dimension %s: %s. Falling back to 768.",
                    self.embedding_dimension,
                    exc,
                )
                self.embedding_dimension = 768
                self.concept_index = ConceptIndex(dimension=self.embedding_dimension)
            for concept in concepts:
                self._add_concept_to_index(concept)

    def _scores_need_recomputation(self, concept: Optional[AlgorithmicConcept]) -> bool:
        if concept is None:
            return False
        scores = concept.scores
        if scores is None or not isinstance(scores, ConceptScores):
            return True
        try:
            values = scores.model_dump()
        except Exception:
            return True
        for value in values.values():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return True
            if not math.isfinite(numeric):
                return True
        try:
            combined = float(concept.combined_score)
        except (TypeError, ValueError):
            return True
        return not math.isfinite(combined)

    def _safe_add_to_db(self, concept: AlgorithmicConcept) -> bool:
        try:
            self.db.add(concept)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to add concept %s to the database: %s",
                concept.id,
                exc,
            )
            return False

    def _safe_update_in_db(self, concept: AlgorithmicConcept) -> bool:
        try:
            self.db.update(concept)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to update concept %s in the database: %s",
                concept.id,
                exc,
            )
            return False

    def _safe_get_all_concepts(self) -> List[AlgorithmicConcept]:
        try:
            concepts = self.db.get_all_programs()
            return [c for c in concepts if c]
        except Exception as exc:
            logger.warning("Error retrieving concepts from the database: %s", exc)
            return []

    def _safe_get_concept(self, concept_id: str) -> Optional[AlgorithmicConcept]:
        try:
            return self.db.get(concept_id)
        except Exception as exc:
            logger.warning(
                "Error fetching concept %s from the database: %s",
                concept_id,
                exc,
            )
            return None

    def _safe_sample(self) -> Tuple[Optional[AlgorithmicConcept], List[AlgorithmicConcept], List[AlgorithmicConcept]]:
        try:
            return self.db.sample()
        except Exception as exc:
            logger.warning("Error sampling population from the database: %s", exc)
            return None, [], []

    def _count_programs(self) -> int:
        try:
            return self.db.count_programs()
        except Exception as exc:
            logger.warning("Failed to count concepts in the database: %s", exc)
            return 0

    def _index_size(self) -> int:
        with self._index_lock:
            try:
                return self.concept_index.size()
            except Exception as exc:
                logger.warning("Failed to obtain FAISS index size: %s", exc)
                return 0

    def _add_concept_to_index(self, concept: Optional[AlgorithmicConcept]) -> None:
        if not concept or not concept.embedding:
            return
        with self._index_lock:
            if len(concept.embedding) != self.embedding_dimension:
                logger.warning(
                    "Skipping concept %s while rebuilding index: expected dimension %s, received %s.",
                    concept.id,
                    self.embedding_dimension,
                    len(concept.embedding),
                )
                return
            try:
                self.concept_index.add_concept(concept.id, concept.embedding)
            except Exception as exc:
                logger.warning("Failed to index concept %s in FAISS: %s", concept.id, exc)

    def _meta_learning_enabled(self) -> bool:
        return self.meta_summarizer is not None

    def _register_evaluated_concept(self, concept: AlgorithmicConcept) -> None:
        if not self._meta_learning_enabled():
            return
        with self._meta_lock:
            self.meta_summarizer.add_evaluated_concept(concept)
            if self.meta_summarizer.should_update_meta(self.meta_interval):
                best_concept = None
                try:
                    best_concept = self.db.get_best_program()
                except Exception as exc:
                    logger.warning("Failed to retrieve best concept for meta-learning: %s", exc)
                recommendations, _ = self.meta_summarizer.update_meta_memory(best_concept)
                if recommendations:
                    self.current_meta_recommendations = recommendations
                    print("  🧠 Meta-learning recommendations refreshed.")
                self._persist_meta_state()

    def _persist_meta_state(self) -> None:
        if not self._meta_learning_enabled():
            return
        with self._meta_lock:
            try:
                self.meta_summarizer.save_state(str(self.meta_state_path))
            except Exception as exc:
                logger.warning("Failed to persist meta-learning state to %s: %s", self.meta_state_path, exc)

    def _initialize_population(self):
        print("🧬 Initializing concept population...")
        max_attempts = self.cfg.evolution.population_size * 3
        attempts = 0
        with tqdm(
            total=self.cfg.evolution.population_size,
            desc="Generating initial concepts",
        ) as pbar:
            while attempts < max_attempts:
                current_size = self._count_programs()
                if current_size >= self.cfg.evolution.population_size:
                    break

                concept = self.idea_generator.generate_initial_concept(
                    self.problem_description, generation=0, model_cfg=self.cfg.model, use_cache=False
                )

                if concept and self._safe_add_to_db(concept):
                    pbar.update(1)
                attempts += 1
        print("  Evaluating initial population...")
        self._evaluate_population()

    def _evaluate_population(self):
        concepts = self._safe_get_all_concepts()
        concepts_to_evaluate = [
            c for c in concepts if self._scores_need_recomputation(c)
        ]
        print(f"  Found {len(concepts_to_evaluate)} concepts to evaluate.")
        index_dirty = False
        inferred_dimension = None

        texts_to_embed = [c.get_full_prompt_text() for c in concepts_to_evaluate]
        if texts_to_embed:
            print("  Generating batched embeddings...")
            all_embeddings = self._get_embedding(texts_to_embed)
            for concept, embedding in zip(concepts_to_evaluate, all_embeddings):
                concept.embedding = embedding

        for concept in tqdm(concepts_to_evaluate, desc="Evaluating concepts"):
            try:
                self._evaluate_single_concept(concept, persist_embedding=False)
                if concept.embedding:
                    inferred_dimension = inferred_dimension or len(concept.embedding)
                index_dirty = True
            except Exception as exc:
                logger.warning(
                    "Failed to evaluate concept %s: %s",
                    getattr(concept, "id", "unknown"),
                    exc,
                )

        if inferred_dimension and inferred_dimension != self.embedding_dimension:
            logger.info(
                "Updating embedding dimension after reevaluation: %s -> %s",
                self.embedding_dimension,
                inferred_dimension,
            )
            self.embedding_dimension = inferred_dimension
        if index_dirty:
            self._rebuild_concept_index()

    def _evaluate_single_concept(
        self, concept: AlgorithmicConcept, *, persist: bool = True, persist_embedding: bool = True
    ) -> None:
        if persist_embedding:
            concept.embedding = self._get_embedding(concept.get_full_prompt_text())

        try:
            raw_scores = self.evaluator.run(concept, model_cfg=self.cfg.model)
        except Exception as exc:
            logger.warning("Error evaluating concept %s: %s", concept.id, exc)
            raw_scores = None

        if raw_scores and not isinstance(raw_scores, ConceptScores):
            try:
                raw_scores = ConceptScores.model_validate(raw_scores)
            except Exception as exc:
                logger.warning(
                    "Invalid scores returned for concept %s: %s",
                    concept.id,
                    exc,
                )
                raw_scores = None

        if raw_scores:
            with self._scoring_lock:
                self.scoring_system.update_history(raw_scores)
                normalized_scores = self.scoring_system.normalize_scores(raw_scores)
            concept.scores = normalized_scores
            try:
                is_aligned, _, alignment_score = self.alignment_validator.run(
                    concept, self.problem_description, model_cfg=self.cfg.model
                )
            except Exception as exc:
                logger.warning("Alignment validator failed for concept %s: %s", concept.id, exc)
                is_aligned = True
                alignment_score = 0.0

            if not is_aligned:
                alignment_score = 0.0

            with self._scoring_lock:
                concept.combined_score = self.scoring_system.calculate_combined_score(
                    concept.scores,
                    concept.generation,
                    self.cfg.evolution.num_generations,
                    alignment_score,
                )
            self._register_evaluated_concept(concept)

        if persist:
            self._safe_update_in_db(concept)

    def _check_novelty_with_faiss(self, draft_concept: AlgorithmicConcept) -> bool:
        if not draft_concept.embedding:
            return True

        with self._index_lock:
            if self.concept_index.size() == 0:
                return True
            try:
                similar_concepts_ids = self.concept_index.find_similar(
                    draft_concept.embedding,
                    k=1,
                    threshold=self.cfg.evolution.novelty_threshold,
                )
            except Exception as exc:
                logger.warning(
                    "Error querying FAISS for concept %s: %s",
                    draft_concept.id,
                    exc,
                )
                return True

        if not similar_concepts_ids:
            return True
        candidate = similar_concepts_ids[0]
        try:
            candidate_id = candidate[0]
        except (TypeError, IndexError):
            logger.warning("Unexpected FAISS index result: %s", candidate)
            return True
        most_similar_concept = self._safe_get_concept(candidate_id)
        if not most_similar_concept:
            logger.info(
                "FAISS returned concept %s, but it was not found in the database.",
                candidate_id,
            )
            return True
        print(f"  🤔 Similar concept detected: '{most_similar_concept.title}'")
        try:
            decision = self.novelty_judge.run(draft_concept, most_similar_concept, model_cfg=self.cfg.model)
        except Exception as exc:
            logger.warning(
                "Novelty judge call failed for concept %s: %s",
                draft_concept.id,
                exc,
            )
            return True
        if decision and not decision.is_novel:
            print(f"  ❌ Discarded as a duplicate: {decision.explanation}")
            return False
        return True

    def _get_verification_rounds(self) -> int:
        configured = getattr(self.cfg.evolution, "verification_retries", None)
        if configured is None:
            return 1
        try:
            rounds = int(configured)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid value for evolution.verification_retries (%s). Falling back to 1 round.",
                configured,
            )
            return 1
        if rounds < 1:
            logger.warning(
                "evolution.verification_retries must be >= 1 (received %s). Defaulting to 1.",
                rounds,
            )
            return 1
        return rounds

    def _execute_verification_loop(self, draft_concept: AlgorithmicConcept) -> AlgorithmicConcept:
        total_rounds = self._get_verification_rounds()
        if not draft_concept.draft_history or draft_concept.draft_history[-1] != draft_concept.description:
            draft_concept.draft_history.append(draft_concept.description)

        for round_idx in range(1, total_rounds + 1):
            print(f"Verification Round {round_idx}/{total_rounds}")
            try:
                report = self.verifier.verify(
                    draft_concept,
                    self.problem_description,
                    model_cfg=self.cfg.model,
                )
            except Exception as exc:
                logger.warning(
                    "Verification agent failed for concept %s in round %s: %s",
                    draft_concept.id,
                    round_idx,
                    exc,
                )
                report = VerificationReport(
                    round_index=round_idx,
                    passed=False,
                    summary=f"Verification error: {exc}",
                    blocking_issues=[f"Agent error: {exc}"],
                )
            else:
                if not isinstance(report, VerificationReport):
                    logger.warning(
                        "Verifier returned unexpected payload for concept %s in round %s: %r",
                        draft_concept.id,
                        round_idx,
                        report,
                    )
                    report = VerificationReport(
                        round_index=round_idx,
                        passed=False,
                        summary="Verifier returned an unexpected payload.",
                        blocking_issues=["Verification agent returned a non-VerificationReport payload."],
                    )
                else:
                    report.round_index = round_idx

            draft_concept.verification_reports.append(report)

            critique_lines = [
                f"Verification Round {round_idx}: {report.summary or 'No summary provided.'}"
            ]
            if report.blocking_issues:
                critique_lines.append("Blocking issues:")
                critique_lines.extend(f"- {issue}" for issue in report.blocking_issues)
            if report.improvement_suggestions:
                critique_lines.append("Suggestions:")
                critique_lines.extend(f"- {suggestion}" for suggestion in report.improvement_suggestions)
            draft_concept.critique_history.append("\n".join(critique_lines))

            if report.passed:
                print("Verification Passed")
                break

            print("Verification Failed")
            if round_idx == total_rounds:
                break

            try:
                corrected_description, applied_fixes = self.solver.correct(
                    draft_concept,
                    report,
                    self.problem_description,
                    model_cfg=self.cfg.model,
                )
            except Exception as exc:
                logger.warning(
                    "Solver failed to correct concept %s in round %s: %s",
                    draft_concept.id,
                    round_idx,
                    exc,
                )
                break

            applied_fixes = list(applied_fixes or [])
            if not corrected_description:
                logger.warning(
                    "Solver returned no correction for concept %s in round %s.",
                    draft_concept.id,
                    round_idx,
                )
                break

            if corrected_description != draft_concept.description:
                draft_concept.description = corrected_description
                draft_concept.draft_history.append(corrected_description)

            if applied_fixes:
                fixes_entry = ["Applied fixes:"]
                fixes_entry.extend(f"- {fix}" for fix in applied_fixes)
                draft_concept.critique_history.append("\n".join(fixes_entry))

        return draft_concept

    def _execute_generation_task(self, generation: int) -> AlgorithmicConcept:
        parent, inspirations, _ = self._safe_sample()
        if not parent:
            raise GenerationSkipped(f"Unable to select a parent for generation {generation}.")

        parent_score = parent.combined_score if isinstance(parent.combined_score, (int, float)) else 0.0
        print(
            f"[Gen {generation}] 👨‍👩‍👧 Selected parent: '{parent.title}' (Score: {parent_score:.2f})"
        )

        try:
            new_concept = self.idea_generator.mutate_or_crossover(
                parent,
                inspirations,
                generation,
                self.problem_description,
                model_cfg=self.cfg.model,
                meta_recommendations=self.current_meta_recommendations,
            )
        except Exception as exc:
            raise GenerationSkipped(
                f"Mutation/crossover failed for generation {generation}: {exc}"
            ) from exc

        if not new_concept:
            raise GenerationSkipped(f"Mutation returned no concept for generation {generation}.")

        new_concept.generation = generation
        new_concept = self._execute_verification_loop(new_concept)
        new_concept.embedding = self._get_embedding(new_concept.get_full_prompt_text())

        if not self._check_novelty_with_faiss(new_concept):
            raise GenerationSkipped(
                f"Concept '{new_concept.title}' for generation {generation} was discarded as a duplicate."
            )

        self._evaluate_single_concept(new_concept, persist=False, persist_embedding=False)
        try:
            new_concept.system_requirements = self.req_extractor.run(new_concept, model_cfg=self.cfg.model)
        except Exception as exc:
            logger.warning(
                "Failed to extract requirements for concept %s: %s",
                new_concept.id,
                exc,
            )

        if not self._safe_add_to_db(new_concept):
            raise RuntimeError(f"Failed to add concept {new_concept.id} to the database.")

        if new_concept.embedding:
            if len(new_concept.embedding) != self.embedding_dimension:
                logger.warning(
                    "Embedding dimension mismatch detected (expected %s, received %s). Rebuilding index.",
                    self.embedding_dimension,
                    len(new_concept.embedding),
                )
                self.embedding_dimension = len(new_concept.embedding)
                self._rebuild_concept_index()
            else:
                self._add_concept_to_index(new_concept)

        score_display = new_concept.combined_score if isinstance(new_concept.combined_score, (int, float)) else 0.0
        print(
            f"[Gen {generation}] ✅ Added concept: '{new_concept.title}' (Score: {score_display:.2f})"
        )

        return new_concept

    def run(self):
        start_gen = 0
        if self.cfg.resume:
            latest_checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            if latest_checkpoint:
                pop, gen, _ = latest_checkpoint
                if pop:
                    for concept in pop:
                        if not self._safe_get_concept(concept.id):
                            self._safe_add_to_db(concept)
                    start_gen = gen
                    print(f"  ✅ Resuming from generation {start_gen}")

        if self._count_programs() == 0:
            self._initialize_population()

        all_concepts = self._safe_get_all_concepts()
        self._rebuild_concept_index(existing_concepts=all_concepts)

        total_generations = int(self.cfg.evolution.num_generations)
        remaining_generations = max(0, total_generations - start_gen)

        self.jobs_submitted = 0
        self.jobs_completed = 0
        self._last_successful_generation = max(self._last_successful_generation, start_gen)

        if remaining_generations == 0:
            self._finalize_run(self._last_successful_generation)
            return

        print(
            f"\n🚀 Starting parallel evolution from generation {start_gen + 1} to {total_generations} "
            f"with {self.max_workers} worker(s).\n"
        )

        next_generation = start_gen + 1
        pending_futures = {}
        interrupted = False

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                while self.jobs_completed < remaining_generations or pending_futures:
                    while (
                        len(pending_futures) < self.max_workers
                        and self.jobs_submitted < remaining_generations
                        and next_generation <= total_generations
                    ):
                        future = executor.submit(self._execute_generation_task, next_generation)
                        pending_futures[future] = next_generation
                        next_generation += 1
                        self.jobs_submitted += 1

                    if not pending_futures:
                        break

                    done, _ = concurrent.futures.wait(
                        list(pending_futures.keys()),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for future in done:
                        gen = pending_futures.pop(future)
                        self.current_generation = gen
                        try:
                            future.result()
                        except GenerationSkipped as skip_exc:
                            print(f"[Gen {gen}] ⚠️ {skip_exc}")
                        except Exception as exc:
                            logger.error("Task for generation %s failed: %s", gen, exc)
                        else:
                            self._last_successful_generation = max(self._last_successful_generation, gen)
                            if gen > 0 and gen % self.db.config.migration_interval == 0:
                                print(f"\n🏝️ Performing island migration triggered by generation {gen}...")
                                try:
                                    population_snapshot = self._safe_get_all_concepts()
                                    self.db.perform_migration(population_snapshot, gen)
                                except Exception as exc:
                                    logger.warning("Error during island migration: %s", exc)
                            if gen % self.cfg.evolution.checkpoint_interval == 0:
                                try:
                                    self.checkpoint_manager.save_checkpoint(
                                        self._safe_get_all_concepts(), gen, self.problem_description
                                    )
                                    self._persist_meta_state()
                                except Exception as exc:
                                    logger.warning(
                                        "Failed to persist checkpoint at generation %s: %s",
                                        gen,
                                        exc,
                                    )
                        finally:
                            self.jobs_completed += 1

                    if self.jobs_completed >= remaining_generations and not pending_futures:
                        break
        except KeyboardInterrupt:
            interrupted = True
            print("\n🛑 Keyboard interrupt received. Waiting for running tasks to finish...")
        finally:
            if pending_futures:
                concurrent.futures.wait(
                    list(pending_futures.keys()),
                    return_when=concurrent.futures.ALL_COMPLETED,
                )

        if interrupted:
            print("  ⚠️ Evolution interrupted before completing all scheduled generations.")

        final_generation = max(self._last_successful_generation, self.db.get_max_generation())
        self._finalize_run(final_generation)

    def _finalize_run(self, final_generation: int) -> None:
        final_generation = max(0, final_generation)
        try:
            population_snapshot = self._safe_get_all_concepts()
            if population_snapshot:
                self.checkpoint_manager.save_checkpoint(
                    population_snapshot,
                    final_generation,
                    self.problem_description,
                )
            self._persist_meta_state()
        except Exception as exc:
            logger.warning("Failed to persist final checkpoint: %s", exc)

        print("\n✅ Evolution complete.")
        try:
            self.db.close()
        except Exception as exc:
            logger.warning("Error closing the database: %s", exc)
