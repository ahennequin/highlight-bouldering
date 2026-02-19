from time import time

from src.video import YoutubeVideoAsset
from src.video.video_wrapper import VideoWrapper


olymics_rings_sequence_frame_timestamps = [
    (
        time(minute=0, second=58, microsecond=800000),
        time(minute=1, second=6, microsecond=500000),
    ),
    (
        time(minute=2, second=51, microsecond=250000),
        time(minute=3, second=0, microsecond=300000),
    ),
    (
        time(minute=4, second=42, microsecond=400000),
        time(minute=5, second=16, microsecond=750000),
    ),
]

original_video_url = "https://www.youtube.com/watch?v=45KmZUc0CzA"


class OlympicRingSequence:
    def __init__(self, start_time: time, end_time: time):
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
        video_sequence = original_video_wrapper.get_frames_from_time(
            self.start_time, self.end_time
        )
        return VideoWrapper.load_video_from_frames(video_sequence)


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
