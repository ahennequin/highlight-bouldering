import argparse
from datetime import datetime
from loguru import logger

from src.features.olympic_rings_detection import (
    OlympicRingsDetector,
    olymics_rings_fade_in_frame_timestamps,
    fade_in_duration,
    olymics_rings_fade_out_frame_timestamps,
    fade_out_duration,
)
from src.metrics.cosine_similarity import CosineSimilarity
from src.video import YoutubeVideoAsset
from src.video import VideoWrapper
from src.video_embedder.clip import ClipVideoEmbedder


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

    # Create two Olympic rings detectors (for fade-in and fade-out sequences)
    fade_in_detector = OlympicRingsDetector(
        start_time_list=olymics_rings_fade_in_frame_timestamps,
        sequence_duration=fade_in_duration,
        video_embedder=ClipVideoEmbedder(),
        metric=CosineSimilarity(),
        threshold=0.79,
    )
    fade_out_detector = OlympicRingsDetector(
        start_time_list=olymics_rings_fade_out_frame_timestamps,
        sequence_duration=fade_out_duration,
        video_embedder=ClipVideoEmbedder(),
        metric=CosineSimilarity(),
        threshold=0.73,
    )

    formatted_date = datetime.today().strftime("%Y%m%d_%H%M")

    # Detect the Olympic rings sequences in the video frames
    df_fade_in = fade_in_detector.detect(video_sequence)
    df_fade_in.to_csv(
        f"./data/embeddings/{formatted_date}_fade_in_predictions.csv", index=False
    )

    df_fade_out = fade_out_detector.detect(video_sequence)
    df_fade_out.to_csv(
        f"./data/embeddings/{formatted_date}_fade_out_predictions.csv", index=False
    )

    # Detect moments where fade-ins and fade-outs are predicted
    ...


if __name__ == "__main__":
    main()
