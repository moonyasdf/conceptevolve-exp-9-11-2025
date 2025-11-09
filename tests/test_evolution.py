from typing import List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from src.concepts import AlgorithmicConcept, ConceptScores, VerificationReport
from src.evolution import ConceptEvolution


class StubConceptIndex:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._entries: List = []
        self.similar_response = []
        self.raise_on_size = False
        self.raise_on_find = False

    def size(self) -> int:
        if self.raise_on_size:
            raise RuntimeError("size failure")
        return len(self._entries)

    def add_concept(self, concept_id: str, embedding: List[float]) -> None:
        self._entries.append((concept_id, embedding))

    def find_similar(
        self,
        embedding: List[float],
        k: int = 5,
        threshold: float = 0.95,
        exclude_id: Optional[str] = None,
    ):
        if self.raise_on_find:
            raise RuntimeError("find failure")
        return self.similar_response


class StubConceptDatabase:
    class _IslandManager:
        def perform_migration(self, *args, **kwargs):
            return None

    def __init__(self, config, initial_concepts=None):
        self.config = config
        self._concepts = list(initial_concepts or [])
        self.island_manager = self._IslandManager()

    def add(self, concept: AlgorithmicConcept) -> None:
        self._replace(concept)

    def update(self, concept: AlgorithmicConcept) -> None:
        self._replace(concept)

    def _replace(self, concept: AlgorithmicConcept) -> None:
        for idx, existing in enumerate(self._concepts):
            if existing.id == concept.id:
                self._concepts[idx] = concept
                return
        self._concepts.append(concept)

    def get(self, concept_id: str) -> Optional[AlgorithmicConcept]:
        for concept in self._concepts:
            if concept.id == concept_id:
                return concept
        return None

    def get_all_programs(self) -> List[AlgorithmicConcept]:
        return list(self._concepts)

    def count_programs(self) -> int:
        return len(self._concepts)

    def sample(self):
        return (self._concepts[0] if self._concepts else None, [], [])

    def get_max_generation(self) -> int:
        if not self._concepts:
            return 0
        return max(concept.generation for concept in self._concepts)

    def close(self) -> None:
        return None


class StubCheckpointManager:
    def __init__(self, *args, **kwargs):
        self.saved = []

    def load_latest_checkpoint(self):
        return [], 0, ""

    def save_checkpoint(self, population, gen, problem_description):
        self.saved.append(gen)


class StubEmbeddingClient:
    def __init__(self, model_name: Optional[str] = None):
        model = model_name or "models/text-embedding-004"
        if not model.startswith("models/"):
            model = f"models/{model}"
        self.model_name = model

    def get_embedding(self, text: str) -> List[float]:
        return [0.0] * 768


class StubSolverAgent:
    def __init__(self):
        self.calls = 0
        self.responses: List[Tuple[Optional[str], Optional[str], List[str]]] = []

    def generate_initial_concept(self, problem_description: str, generation: int, model_cfg=None, use_cache=True):
        return None

    def mutate_or_crossover(self, parent, inspirations, generation, problem_description, model_cfg=None, meta_recommendations=None):
        return None
    
    def correct(self, concept, report, problem_description, model_cfg=None, persona=None):
        self.calls += 1
        if self.responses:
            title, description, change_log = self.responses.pop(0)
            if title:
                concept.title = title
            if description and description != concept.description:
                concept.description = description
                if not concept.draft_history or concept.draft_history[-1] != description:
                    concept.draft_history.append(description)
            if change_log and concept.verification_reports:
                applied = "\n".join(f"- {item}" for item in change_log)
                latest = concept.verification_reports[-1]
                latest.diagnostics = (
                    f"Applied fixes:\n{applied}" if not latest.diagnostics else f"{latest.diagnostics}\n\nApplied fixes:\n{applied}"
                )
        return concept


class StubConceptEvaluator:
    def run(self, concept) -> ConceptScores:
        return ConceptScores(novelty=5.0, potential=5.0, sophistication=5.0, feasibility=5.0)


class StubNoveltyJudge:
    class Decision:
        is_novel = True
        explanation = ""

    def run(self, draft, existing):
        return self.Decision()


class StubRequirementsExtractor:
    def run(self, concept):
        return concept.system_requirements


class StubAlignmentValidator:
    def run(self, concept, problem_description):
        return True, None, 10.0


class StubVerifierAgent:
    def __init__(self):
        self.calls = 0
        self.reports: List[VerificationReport] = []

    def verify(self, concept, problem_description, model_cfg=None):
        self.calls += 1
        if self.reports:
            return self.reports.pop(0)
        return VerificationReport(round_index=self.calls, passed=True, issue_summaries=[])


class StubSolverAgentForVerification:
    def __init__(self):
        self.calls = 0
        self.responses: List[Tuple[Optional[str], Optional[str], List[str]]] = []

    def correct(self, concept, report, problem_description, model_cfg=None, persona=None):
        self.calls += 1
        if self.responses:
            title, description, change_log = self.responses.pop(0)
            if title:
                concept.title = title
            if description and description != concept.description:
                concept.description = description
                if not concept.draft_history or concept.draft_history[-1] != description:
                    concept.draft_history.append(description)
            if change_log and concept.verification_reports:
                applied = "\n".join(f"- {item}" for item in change_log)
                latest = concept.verification_reports[-1]
                latest.diagnostics = (
                    f"Applied fixes:\n{applied}" if not latest.diagnostics else f"{latest.diagnostics}\n\nApplied fixes:\n{applied}"
                )
        return concept


def make_config(tmp_path) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {"embedding_model": "text-embedding-004"},
            "evolution": {
                "population_size": 1,
                "num_generations": 1,
                "novelty_threshold": 0.9,
                "verification_retries": 2,
                "checkpoint_interval": 5,
            },
            "database": {
                "db_path": str(tmp_path / "test.db"),
                "num_islands": 1,
                "archive_size": 1,
                "migration_interval": 5,
                "migration_rate": 0.1,
                "island_elitism": False,
                "parent_selection_strategy": "weighted",
                "parent_selection_lambda": 1.0,
                "exploitation_alpha": 1.0,
                "exploitation_ratio": 0.5,
                "num_archive_inspirations": 0,
                "num_top_k_inspirations": 0,
                "enforce_island_separation": False,
            },
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "resume": False,
        }
    )


def make_evolution(monkeypatch, tmp_path, initial_concepts=None):
    initial_concepts = list(initial_concepts or [])
    created = {}

    def db_factory(config):
        db = StubConceptDatabase(config, initial_concepts=initial_concepts)
        created["db"] = db
        return db

    def verifier_factory():
        stub = StubVerifierAgent()
        created["verifier"] = stub
        return stub

    def solver_factory():
        stub = StubSolverAgent()
        created["solver"] = stub
        return stub

    monkeypatch.setattr("src.evolution.ConceptDatabase", db_factory)
    monkeypatch.setattr("src.evolution.CheckpointManager", StubCheckpointManager)
    monkeypatch.setattr("src.evolution.EmbeddingClient", StubEmbeddingClient)
    monkeypatch.setattr("src.evolution.ConceptIndex", StubConceptIndex)
    monkeypatch.setattr("src.evolution.SolverAgent", solver_factory)
    monkeypatch.setattr("src.evolution.ConceptEvaluator", StubConceptEvaluator)
    monkeypatch.setattr("src.evolution.NoveltyJudge", StubNoveltyJudge)
    monkeypatch.setattr("src.evolution.RequirementsExtractor", StubRequirementsExtractor)
    monkeypatch.setattr("src.evolution.AlignmentValidator", StubAlignmentValidator)
    monkeypatch.setattr("src.evolution.VerifierAgent", verifier_factory)

    cfg = make_config(tmp_path)
    evo = ConceptEvolution("Problema", cfg)
    created.setdefault("verifier", evo.verifier)
    created.setdefault("solver", evo.solver)
    created.setdefault("db", created.get("db"))
    return evo, created


def test_concept_index_uses_configured_embedding_dimension(monkeypatch, tmp_path):
    evo, _ = make_evolution(monkeypatch, tmp_path)
    assert evo.embedding_dimension == 768
    assert evo.concept_index.dimension == 768


def test_evaluate_population_recomputes_invalid_scores(monkeypatch, tmp_path):
    concept = AlgorithmicConcept(title="Corrupto", description="desc")
    concept.scores = ConceptScores(
        novelty=float("nan"), potential=5.0, sophistication=5.0, feasibility=5.0
    )
    concept.combined_score = float("nan")

    evo, _ = make_evolution(monkeypatch, tmp_path, initial_concepts=[concept])

    call_counter = {"count": 0}

    def fake_eval(self, concept, persist=True, persist_embedding=True):
        call_counter["count"] += 1
        concept.scores = ConceptScores(
            novelty=5.0, potential=5.0, sophistication=5.0, feasibility=5.0
        )
        concept.combined_score = 5.0

    monkeypatch.setattr(ConceptEvolution, "_evaluate_single_concept", fake_eval)

    evo._evaluate_population()

    assert call_counter["count"] == 1


def test_check_novelty_handles_missing_db_result(monkeypatch, tmp_path):
    evo, _ = make_evolution(monkeypatch, tmp_path)
    evo.concept_index._entries.append(("existing", [0.0] * evo.embedding_dimension))
    evo.concept_index.similar_response = [("missing-id", 0.99)]

    draft = AlgorithmicConcept(title="Draft", description="desc")
    draft.embedding = [0.1] * evo.embedding_dimension

    assert evo._check_novelty_with_faiss(draft) is True


def test_check_novelty_handles_faiss_errors(monkeypatch, tmp_path):
    evo, _ = make_evolution(monkeypatch, tmp_path)
    evo.concept_index._entries.append(("existing", [0.0] * evo.embedding_dimension))
    evo.concept_index.raise_on_find = True

    draft = AlgorithmicConcept(title="Draft", description="desc")
    draft.embedding = [0.1] * evo.embedding_dimension

    assert evo._check_novelty_with_faiss(draft) is True


def test_verification_loop_handles_fail_then_pass(monkeypatch, tmp_path):
    evo, created = make_evolution(monkeypatch, tmp_path)
    verifier: StubVerifierAgent = created["verifier"]
    solver: StubSolverAgent = created["solver"]

    concept = AlgorithmicConcept(title="Test", description="initial draft")

    verifier.reports = [
        VerificationReport(round_index=0, passed=False, issue_summaries=["issue"]),
        VerificationReport(round_index=0, passed=True, issue_summaries=[]),
    ]
    solver.responses = [("Updated Title", "corrected draft", ["Addressed issue"])]

    evo.cfg.evolution.verification_retries = 3

    updated = evo._execute_verification_loop(concept)

    assert verifier.calls == 2
    assert solver.calls == 1
    assert updated.description == "corrected draft"
    assert len(updated.verification_reports) == 2
    assert updated.verification_reports[0].round_index == 1
    assert updated.verification_reports[1].passed is True
    assert len(updated.draft_history) == 2
    first_report = updated.verification_reports[0]
    assert first_report.diagnostics and "Addressed issue" in first_report.diagnostics


def test_verification_loop_defaults_to_single_round(monkeypatch, tmp_path):
    evo, created = make_evolution(monkeypatch, tmp_path)
    verifier: StubVerifierAgent = created["verifier"]
    solver: StubSolverAgent = created["solver"]

    concept = AlgorithmicConcept(title="Test", description="initial draft")

    if "verification_retries" in evo.cfg.evolution:
        del evo.cfg.evolution["verification_retries"]

    verifier.reports = [VerificationReport(round_index=0, passed=True, issue_summaries=[])]

    updated = evo._execute_verification_loop(concept)

    assert verifier.calls == 1
    assert solver.calls == 0
    assert len(updated.verification_reports) == 1
    assert updated.verification_reports[0].passed is True
    assert len(updated.draft_history) == 1
