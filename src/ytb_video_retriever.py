from pytubefix import YouTube


class YoutubeVideoRetriever:
    """A class to retrieve YouTube video information and streams."""

    def __init__(self):
        pass

    def retrieve_video(self, video_url) -> tuple[YouTube, dict]:
        yt = YouTube(video_url)
        video_info = {
            "title": yt.title,
            "author": yt.author,
            "length": yt.length,
            "views": yt.views,
            "description": yt.description,
        }
        return yt, video_info


class YoutubeVideoAsset:
    """A class to represent a YouTube video asset with its stream and metadata."""

    def __init__(self, video_url):
        self.video_url = video_url
        self.video, self.video_info = YoutubeVideoRetriever().retrieve_video(video_url)

    @property
    def stream(self):
        return self.video.streams.get_highest_resolution()

    @property
    def codec(self) -> str:
        return self.stream.codecs[0]

    @property
    def fps(self) -> int:
        return self.stream.fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self.stream.width, self.stream.height
