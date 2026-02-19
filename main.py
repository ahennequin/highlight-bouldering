import argparse
from loguru import logger

from src.ytb_video_retriever import YoutubeVideoRetriever


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

    video, video_info = YoutubeVideoRetriever().retrieve_video(args.url)
    logger.info(f"Retrieved video info: {video_info}")


if __name__ == "__main__":
    main()
