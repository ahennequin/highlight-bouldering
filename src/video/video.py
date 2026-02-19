from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pytubefix import YouTube
from pytubefix import Stream


@dataclass
class VideoInfo:
    title: str
    author: str
    length: int
    views: int
    description: str


class YoutubeVideoRetriever:
    """A class to retrieve YouTube video information and streams."""

    def __init__(self):
        pass

    def retrieve_video(self, video_url) -> tuple[YouTube, VideoInfo]:
        yt = YouTube(video_url)
        video_info = VideoInfo(
            title=yt.title,
            author=yt.author,
            length=yt.length,
            views=yt.views,
            description=yt.description,
        )
        return yt, video_info


class YoutubeVideoAsset:
    """A class to represent a YouTube video asset with its stream and metadata."""

    def __init__(self, video_url):
        self.video_url = video_url
        self.video, self.video_info = YoutubeVideoRetriever().retrieve_video(video_url)

    @property
    def stream(self) -> Stream:
        return self.video.streams.get_highest_resolution()

    @property
    def codec(self) -> str:
        return self.stream.codecs[0]

    @property
    def fps(self) -> float:
        return self.stream.fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self.stream.width, self.stream.height

    def download(self, output_path: str):
        if not (path := Path(output_path)).parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if not Path(output_path).exists():
            self.stream.download(output_path=path.parent, filename=path.name)
        else:
            logger.warning(f"File {output_path} already exists. Skipping download.")
