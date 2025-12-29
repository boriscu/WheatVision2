# WheatVision2

SAM vs SAM2 Segmentation Comparison Tool for wheat phenotyping.

## Overview

WheatVision2 is a modular tool for comparing SAM (Segment Anything Model) and SAM2 segmentation models on wheat ear detection in video recordings. It provides:

- **SAM Mode**: Frame-by-frame segmentation with grid-based prompts
- **SAM2 Mode**: Video propagation with memory-based tracking
- **Preprocessing**: Background removal and ROI detection for wheat images
- **Postprocessing**: Wheat ear filtering based on aspect ratio and size
- **Metrics**: Speed (FPS, timing) and accuracy (temporal consistency) comparison
- **Export**: Masks (PNG/NPY), videos (MP4), reports (JSON/CSV)

## Installation

### 1. Clone this repository

```bash
git clone <repo_url>
cd WheatVision2
```

### 2. Install dependencies

```bash
conda activate wheatvision
pip install -r requirements.txt
```

### 3. Clone & install SAM (editable)

```bash
git clone https://github.com/facebookresearch/segment-anything.git external/sam_repo
pip install -e external/sam_repo
```

Download checkpoint:
```bash
cd external/sam_repo
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h.pth
cd ../..
```

### 4. Clone & install SAM2 (editable)

```bash
git clone https://github.com/facebookresearch/sam2.git external/sam2_repo
pip install -e external/sam2_repo
```

Download checkpoint:
```bash
cd external/sam2_repo/checkpoints && bash download_ckpts.sh
cd ../../..
```

### 5. Configure environment

Copy the example environment file and adjust paths if needed:

```bash
cp .env.example .env
```

Edit `.env` to match your setup:

```bash
WHEATVISION_SAM_REPO=external/sam_repo
WHEATVISION_SAM_CHECKPOINT=external/sam_repo/checkpoints/sam_vit_h.pth
WHEATVISION_SAM2_REPO=external/sam2_repo
WHEATVISION_SAM2_CFG=external/sam2_repo/sam2/configs/sam2.1/sam2.1_hiera_s.yaml
WHEATVISION_SAM2_CKPT=external/sam2_repo/checkpoints/sam2.1_hiera_small.pt
```

## Usage

### Launch the Gradio Interface

```bash
conda activate wheatvision && python -m wheatvision.ui.app
```

Then open http://127.0.0.1:7860 in your browser.

### Programmatic Usage

```python
from wheatvision.config.constants import SegmentationModel
from wheatvision.pipeline import SegmentationPipeline

# Create SAM pipeline
pipeline = SegmentationPipeline(SegmentationModel.SAM)

# Process video
frames, preprocess, results, metrics = pipeline.process_video("path/to/video.mp4")

# Print metrics
print(f"FPS: {metrics.speed_metrics.fps}")
print(f"Temporal Consistency: {metrics.accuracy_metrics.temporal_consistency_score}")
```

## Project Structure

```
WheatVision2/
├── wheatvision/
│   ├── config/          # Settings, constants, data models
│   ├── io/              # Frame/video I/O
│   ├── preprocessing/   # Background removal, ROI detection
│   ├── engines/         # SAM and SAM2 wrappers
│   ├── postprocessing/  # Wheat ear filtering
│   ├── metrics/         # Speed and accuracy metrics
│   ├── export/          # Masks, video, report exporters
│   ├── pipeline/        # End-to-end processing
│   └── ui/              # Gradio interface
├── external/            # Cloned SAM/SAM2 repos (not committed)
├── exports/             # Output files
├── .env.example         # Configuration template
└── requirements.txt     # Python dependencies
```

## Configuration

All settings are loaded from `.env` file:

| Variable | Description |
|----------|-------------|
| `WHEATVISION_SAM_*` | SAM model paths and settings |
| `WHEATVISION_SAM2_*` | SAM2 model paths and settings |
| `WHEATVISION_BG_HSV_*` | Background HSV thresholds |
| `WHEATVISION_EAR_*` | Wheat ear filter parameters |
| `WHEATVISION_UI_*` | Gradio interface settings |

## License

MIT