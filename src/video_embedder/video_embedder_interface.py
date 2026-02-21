import numpy as np

from src.video.video_wrapper import VideoWrapper


class VideoEmbedderInterface:
    def embed(self, video_sequence: VideoWrapper) -> np.ndarray:
        raise NotImplementedError("This method should be implemented by subclasses.")
