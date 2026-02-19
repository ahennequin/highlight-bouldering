from datetime import time

import cv2


class VideoWrapper:
    """OpenCV wrapper to abstract away OpenCV specific code from the rest of the application."""

    @staticmethod
    def load_video_from_path(video_path: str):
        """Load a video from a given file path."""
        return VideoWrapper(video_path)

    @staticmethod
    def load_video_from_frames(frames: list):
        """Load a video from a list of frames."""
        # Create a temporary video file from the frames
        temp_video_path = "./temp_video.mp4"
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        return VideoWrapper(temp_video_path)

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

    def __len__(self) -> int:
        """Get the total number of frames in the video."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_position(self) -> int:
        """Get the current frame position in the video."""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    @frame_position.setter
    def frame_position(self, frame_index: int) -> None:
        """Set the current frame position in the video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def get_frames_from_time(self, start_time: time, end_time: time = None) -> list:
        """Get frames from the video starting from a specific time (in seconds)."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_index_start = (
            (start_time.microsecond / 1e6) + start_time.second + start_time.minute * 60
        ) * fps
        frame_index_end = (
            ((end_time.microsecond / 1e6) + end_time.second + end_time.minute * 60)
            * fps
            if end_time is not None
            else None
        )
        return self.get_frames_from_indexes(frame_index_start, frame_index_end)

    def get_frames_from_indexes(
        self, frame_index_start: int, frame_index_end: int = None
    ) -> list:
        """Get frames from the video starting from a specific index."""
        self.frame_position = frame_index_start
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:  # Break if we have reached the end of the video
                break
            frames.append(frame)
            # Break if we have reached the end frame index (if specified)
            if frame_index_end is not None and self.frame_position >= frame_index_end:
                break

        return frames

    def __iter__(self):
        return self

    def __next__(self):
        """Read the next frame from the video. Returns None when the video ends."""
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return frame

    def __exit__(self, exc_type, exc, tb) -> None:
        """Release the video capture when done."""
        self.cap.release()

    def __del__(self) -> None:
        """Ensure the video capture is released when the object is deleted."""
        self.cap.release()
