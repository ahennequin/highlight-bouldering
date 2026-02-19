import cv2


class VideoWrapper:
    """OpenCV wrapper to abstract away OpenCV specific code from the rest of the application."""

    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

    def __len__(self):
        """Get the total number of frames in the video."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def frame_position(self) -> int:
        """Get the current frame position in the video."""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    @frame_position.setter
    def frame_position(self, frame_index: int):
        """Set the current frame position in the video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    def get_frames(self, frame_index_start: int, frame_index_end: int = None):
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

    def __exit__(self, exc_type, exc, tb):
        """Release the video capture when done."""
        self.cap.release()

    def __del__(self):
        """Ensure the video capture is released when the object is deleted."""
        self.cap.release()
