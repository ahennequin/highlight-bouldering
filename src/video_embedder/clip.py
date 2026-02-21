import clip
from PIL import Image
import numpy as np
from src.video import VideoWrapper
from src.video_embedder.video_embedder_interface import VideoEmbedderInterface


class ClipVideoEmbedder(VideoEmbedderInterface):
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):

        self.model, self.preprocess = clip.load(model_name, device=device)

    def embed(self, video_sequence: VideoWrapper) -> np.ndarray:
        # Preprocess and embed each frame
        embeddings = []
        for frame in video_sequence:
            image = Image.fromarray(frame)
            image = self.preprocess(image)
            embedding = (
                self.model.encode_image(image.unsqueeze(0)).detach().cpu().numpy()
            )
            embeddings.append(embedding)

        return np.array(embeddings)
