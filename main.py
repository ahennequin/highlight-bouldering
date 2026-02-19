import argparse
from loguru import logger

from src.video import YoutubeVideoAsset
from src.video import VideoWrapper


def main():
    parser = argparse.ArgumentParser(
        description="Run the YouTube video retriever on a given URL."
    )
    parser.add_argument(
        "url", type=str, help="The URL of the YouTube video to retrieve."
    )
    args = parser.parse_args()

    if not args.url:
        logger.error("Please provide a URL.")

    # Create video information and download the video
    yt_asset = YoutubeVideoAsset(args.url)
    logger.info(
        f"Retrieved video info: {yt_asset.video_info.title} by {yt_asset.video_info.author}"
    )

    video_download_path = f"./data/raw/{yt_asset.video_info.title}.mp4"
    yt_asset.download(output_path=video_download_path)

    # Transform the video into a sequence of frames
    video_sequence = VideoWrapper(video_download_path)
    logger.info(f"Extracted {len(video_sequence)} frames from the video.")


if __name__ == "__main__":
    main()
