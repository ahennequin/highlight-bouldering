from datetime import timedelta
from typing import final

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.metrics.metric_interface import MetricInterface
from src.video import YoutubeVideoAsset
from src.video.video_wrapper import VideoWrapper
from src.video_embedder.video_embedder_interface import VideoEmbedderInterface

# Constants for the Olympic rings detection
olymics_rings_fade_in_frame_timestamps: final = [
    timedelta(minutes=0, seconds=58, microseconds=800000),
    timedelta(minutes=2, seconds=51, microseconds=250000),
    timedelta(minutes=4, seconds=42, microseconds=400000),
]

olymics_rings_fade_out_frame_timestamps: final = [
    timedelta(minutes=1, seconds=6, microseconds=500000),
    timedelta(minutes=3, seconds=0, microseconds=300000),
    timedelta(minutes=5, seconds=16, microseconds=750000),
]

fade_in_duration: final = timedelta(seconds=1)
fade_out_duration: final = timedelta(seconds=1)

original_video_url = "https://www.youtube.com/watch?v=45KmZUc0CzA"


class OlympicRingSequence:
    """Class representing a video sequence corresponding to the Olympic rings in the original video."""

    def __init__(self, start_time: timedelta, end_time: timedelta):
        self.start_time = start_time
        self.end_time = end_time

    def extract_sequence_from_video(
        self, video_url: str = original_video_url
    ) -> VideoWrapper:
        """Extract the video sequence corresponding to the Olympic rings from the original video."""
        # Download the original video
        original_video_asset = YoutubeVideoAsset(video_url)
        original_video_asset.download(
            f"./data/raw/{original_video_asset.video_info.title}.mp4"
        )
        # Create a video wrapper for the original video
        original_video_wrapper = VideoWrapper(
            f"./data/raw/{original_video_asset.video_info.title}.mp4"
        )
        # Extract the frames corresponding to the Olympic rings sequence
        frames_list = original_video_wrapper.get_frames_from_time(
            self.start_time, self.end_time
        )

        logger.info(f"Retrieved {len(frames_list)} frames for Olympic Sequence")

        return VideoWrapper.load_video_from_frames(
            frames_list,
            original_video_asset.fps,
        )


class OlympicRingsDetector:
    """Class representing the Olympic rings detector, which is bootstrapped using the video sequences corresponding to the Olympic rings in the original video."""

    def __init__(
        self,
        start_time_list: list[timedelta],
        sequence_duration: list[timedelta],
        video_embedder: VideoEmbedderInterface,
        metric: MetricInterface,
        threshold: float,
    ):
        self.sequence_duration = sequence_duration
        self.video_embedder = video_embedder
        self.metric = metric
        self.threshold = threshold
        self.discriminator_sequences = []

        for start_time in start_time_list:
            self.discriminator_sequences.append(
                OlympicRingSequence(start_time, start_time + sequence_duration)
            )

        self.discriminator_embeddings = self.bootstrap()

    def bootstrap(self):
        """Bootstrap the Olympic rings detector by extracting the video sequences corresponding to the Olympic rings.
        Then, embed the video sequences using the video embedder and average the embeddings to get a single embedding representing the Olympic rings.
        """
        video_sequences = []
        for sequence in self.discriminator_sequences:
            video_sequences.append(sequence.extract_sequence_from_video())

        sequences_embeddings = [
            self.video_embedder.embed(sequence) for sequence in video_sequences
        ]

        return np.array(sequences_embeddings).mean(axis=0)

    def _detect_sequence(self, sequence: VideoWrapper) -> bool:
        """Detect Olympic rings in the given sequence of frame (of the same size as the discriminator)."""

        # Embed the frame using the video embedder
        sequence_embedding = self.video_embedder.embed(sequence)

        # Discriminator and given sequence embeddings must have similar shapes
        assert (
            sequence_embedding.shape == self.discriminator_embeddings.shape
        ), f"Embeddings have different shapes ({sequence_embedding.shape} while expecting {self.discriminator_embeddings.shape})"

        # Compute the similarity between the frame embedding and the discriminator embedding using the metric
        similarity = self.metric.compute(
            sequence_embedding, self.discriminator_embeddings
        )

        return similarity

    def detect(self, video: VideoWrapper) -> pd.DataFrame:
        """Detect Olympic rings on a video of any number of frames greater than the discriminator's."""

        stride = self.discriminator_embeddings.shape[0]

        # For performance reasons, retrieve all frames at once (might be memory consuming)
        full_video_frames = video.get_frames_from_indexes(0, len(video))

        score = []
        frame_idx = []
        for start_frame in tqdm(
            range(0, len(video) - stride - 1, stride),
            desc="Dectecting Olympic Rings...",
            unit="sequence",
        ):
            sequence = full_video_frames[start_frame : start_frame + stride]

            frame_idx.append(start_frame)
            score.append(self._detect_sequence(sequence))

        return pd.DataFrame(
            {
                "frame_idx": frame_idx,
                "score": score,
                "detect": map(lambda x: x > self.threshold, score),
            }
        )
