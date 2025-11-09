"""FAISS-backed vector index for similarity search across concepts."""

from typing import List, Optional, Tuple

import faiss
import numpy as np


class ConceptIndex:
    """Approximate nearest-neighbour search over concept embeddings."""

    def __init__(self, dimension: int = 768) -> None:
        """Create an inner-product FAISS index for L2-normalised vectors."""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.concept_ids: List[str] = []
        self.concept_map: dict[str, int] = {}

    def add_concept(self, concept_id: str, embedding: List[float]) -> None:
        """Add a concept embedding to the index."""
        if not embedding or len(embedding) != self.dimension:
            print(f"  ⚠️ Invalid embedding for concept {concept_id}")
            return

        embedding_np = np.array([embedding], dtype="float32")
        faiss.normalize_L2(embedding_np)

        idx = len(self.concept_ids)
        self.index.add(embedding_np)
        self.concept_ids.append(concept_id)
        self.concept_map[concept_id] = idx

    def find_similar(
        self,
        embedding: List[float],
        k: int = 5,
        threshold: float = 0.95,
        exclude_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Return up to ``k`` concepts whose cosine similarity exceeds ``threshold``."""
        if not embedding or len(embedding) != self.dimension:
            return []
        if not self.concept_ids:
            return []

        query = np.array([embedding], dtype="float32")
        faiss.normalize_L2(query)

        search_k = min(k + 1, len(self.concept_ids))
        distances, indices = self.index.search(query, search_k)

        results: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.concept_ids):
                continue
            concept_id = self.concept_ids[idx]
            similarity = float(dist)
            if similarity >= threshold and concept_id != exclude_id:
                results.append((concept_id, similarity))

        return results[:k]

    def get_all_embeddings_matrix(self) -> np.ndarray:
        """Return a matrix of all embeddings stored inside the FAISS index."""
        if not self.concept_ids:
            return np.array([])
        embeddings = faiss.rev_swig_ptr(
            self.index.xb, self.index.ntotal * self.dimension
        )
        return embeddings.reshape(self.index.ntotal, self.dimension)

    def size(self) -> int:
        """Return the number of indexed concepts."""
        return len(self.concept_ids)
