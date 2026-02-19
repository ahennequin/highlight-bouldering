import numpy as np

from src.metrics.metric_interface import MetricInterface


class CosineSimilarity(MetricInterface):
    def compute(self, frame_embedding_a, frame_embedding_b) -> float:
        """Compute the cosine similarity between two frame embeddings."""
        # Compute cosine similarity between the two frame embeddings
        dot_product = np.dot(frame_embedding_a, frame_embedding_b)
        norm_frame_a = np.linalg.norm(frame_embedding_a)
        norm_frame_b = np.linalg.norm(frame_embedding_b)
        if norm_frame_a == 0 or norm_frame_b == 0:
            return 0
        cosine_similarity = dot_product / (norm_frame_a * norm_frame_b)
        return cosine_similarity
