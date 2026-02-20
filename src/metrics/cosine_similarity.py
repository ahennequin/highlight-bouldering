import numpy as np

from src.metrics.metric_interface import MetricInterface


class CosineSimilarity(MetricInterface):
    def compute(self, frame_embedding_a, frame_embedding_b) -> float:
        """Compute the cosine similarity between two frame embeddings."""
        normalized_embeddings_a = frame_embedding_a / (
            np.linalg.norm(frame_embedding_a, axis=-1, keepdims=True) + 1e-8
        )
        normalized_embeddings_b = frame_embedding_b / (
            np.linalg.norm(frame_embedding_b, axis=-1, keepdims=True) + 1e-8
        )

        cosine_similarity = np.einsum(
            "bnd,bmd->bnm",
            normalized_embeddings_a,
            normalized_embeddings_b,
        )

        return cosine_similarity.mean()
