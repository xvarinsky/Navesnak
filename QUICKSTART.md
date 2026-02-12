# Magic Mirror Face Filter - Quick Start Guide

## Quick Start

```bash
# On Raspberry Pi ARM64 (Debian 13+)
sudo apt update
sudo apt install -y python3-opencv python3-numpy
pip3 install --user -r requirements.txt

# Run
python3 main.py --windowed
```

## Controls

| Key | Action |
|---------|-------|
| **SPACE** | Next filter |
| **B** | Previous filter |
| **1-4** | Select filter directly |
| **C** | Capture screenshot |
| **F** | Toggle fullscreen |
| **Q** / **ESC** | Quit |

## Project Structure

```
Navesnak/
├── main.py           # Main application
├── face_detector.py  # OpenCV DNN face detection
├── filters.py        # Filter configuration and overlay logic
├── requirements.txt  # Python dependencies
├── models/           # DNN models (downloaded automatically)
└── assets/filters/   # PNG filter images
```

## How It Works

### 1. Initialization
- `main.py` -> `MagicMirrorApp.__init__()` initializes camera and face detector

### 2. Face Detection
- `face_detector.py` -> OpenCV DNN SSD model detects face
- `FacemarkLBF` extracts 68 landmarks (eyes, nose, forehead, mouth)
- Detection runs every 3rd frame for performance

### 3. Filter Application
- `filters.py` -> `FilterManager` loads PNG images with alpha channels
- `overlay_filter()` applies filter at anchor point (nose, eyes, forehead)
- Vectorized alpha blending for fast compositing

### 4. Main Loop
```
while True:
    1. Read frame from camera
    2. Mirror flip (cv2.flip)
    3. Detect face and landmarks (every 3rd frame)
    4. Apply current filter overlay
    5. Draw UI (carousel, face indicator)
    6. Display frame
    7. Handle keyboard input (SPACE/B/Q/C/F)
```

## Available Filters

| Filter | Anchor Point | Description |
|--------|-------------|-------------|
| Golden Crown | forehead | Crown sitting on top of head |
| Butterfly Wings | nose | Wings spread around the face |
| Fire Eyes | eyes_center | Fire effect over the eyes |
| Angel Halo | forehead | Halo floating above the head |

## Adding a New Filter

1. Add a 500x500 RGBA PNG image to `assets/filters/`
2. Edit `filters.py`, add to `filter_defs`:
```python
{
    "name": "My Filter",
    "image": "my_filter.png",
    "anchor": "forehead",  # nose, eyes_center, forehead, mouth
    "scale_factor": 1.0,   # size relative to face width
    "offset_y": -50,       # vertical offset (negative = up)
}
```

## Troubleshooting

**Camera not working:**
```bash
ls /dev/video*
python3 main.py --camera-index 1
```

**Models not downloaded:**
```bash
mkdir -p models && cd models
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
wget https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
```
