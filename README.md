# Bouldering Highlight Detection Exercise

This project is an exercise in which a solution is designed to automatically detect highlights in bouldering videos.

## Overview

The goal of this exercise is to create a system that can automatically detect highlights in bouldering videos. Highlights in this context refer to moments in the video that are particularly exciting or significant, such as successful climbs, falls, or other notable events.

## Approach

The solution involves several steps:

1. **Video Processing**: The video is processed frame by frame to extract relevant features.
2. **Feature Extraction**: Use of Computer Vision to extract meaningful features from each frames.
3. **Highlight Detection**: Using the extracted features, the system identifies potential highlights.
4. **Post-processing**: The detected highlights are refined and formatted for output.

## Usage

To use this project, follow these steps:

1. Clone the repository:

    ```git clone git@github.com:ahennequin/hightlight-bouldering.git```

2. Get into project workspace

    ```cd highligh-bouldering```

3. Set up environment

    ```uv sync```

4. Play around with the file in `./notebooks`