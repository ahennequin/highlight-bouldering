from datetime import timedelta
from typing import final

from loguru import logger
import numpy as np

from src.video import YoutubeVideoAsset
from src.video.video_wrapper import VideoWrapper

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

        return VideoWrapper.load_video_from_frames(frames_list)


class OlympicRingsDetector:
    def __init__(self, video_embedder):
        self.video_embedder = video_embedder
        self.sequences = []
        for start_time, end_time in olymics_rings_sequence_frame_timestamps:
            self.sequences.append(OlympicRingSequence(start_time, end_time))

        self.discriminator_embeddings = self.bootstrap()

    def bootstrap(self):
        """Bootstrap the Olympic rings detector by extracting the video sequences corresponding to the Olympic rings.
        Then, embed the video sequences using the video embedder and average the embeddings to get a single embedding representing the Olympic rings.
        """
        video_sequences = []
        for sequence in self.sequences:
            video_sequences.append(sequence.extract_sequence_from_video())

        sequences_embeddings = self.video_embedder(video_sequences)

        return sequences_embeddings.mean(axis=0)

    def detect(self, frame):
        """Detect Olympic rings in the given video frame."""

        return []
