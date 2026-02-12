# Magic Mirror Face Filter

A fun, interactive Raspberry Pi application that uses a camera to show a live mirrored video with face detection and overlay filters (crown, butterfly wings, fire eyes, angel halo).

## Features

- **Live Video Mirror**: Real-time mirrored video feed from camera
- **Face Detection**: OpenCV DNN-based face detection (ARM64 compatible)
- **Facial Landmarks**: 68-point facial landmark detection for precise filter positioning
- **Multiple Filters**: Switch between different fun filters
- **Fullscreen Display**: Optimized for HDMI display on Raspberry Pi
- **Keyboard Controls**: Easy filter switching and quit functionality
- **Transparent Overlays**: PNG filters with alpha blending
- **Automatic Model Download**: Required models are downloaded on first run
- **Frame Skipping**: Efficient detection — runs only every 3rd frame to reduce CPU load

## Hardware Requirements

- Raspberry Pi 4/5 (tested on ARM64/aarch64)
- Camera: Raspberry Pi Camera Module or USB webcam
- Display: External monitor connected via HDMI
- Raspberry Pi OS / Debian GNU/Linux 13+ with Python 3.11+

## Installation

### Quick Start (Raspberry Pi ARM64)

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install system dependencies
sudo apt install -y python3-pip python3-opencv python3-numpy

# 3. Install Python dependencies
pip3 install --user opencv-contrib-python numpy requests

# 4. Run the application
python3 main.py --windowed  # Test in windowed mode first
```

### Alternative: Using Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python3 main.py
```

### Camera Setup

**Raspberry Pi Camera Module:**
```bash
# Enable camera in raspi-config
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
sudo reboot
```

**USB Webcam:** Works out of the box, usually at index 0.

### Add Filter Images

Place your filter PNG images (500x500 RGBA with transparency) in `assets/filters/`:
- `crown.png`
- `butterfly.png`
- `eyes.png`
- `halo.png`

See `assets/filters/README.md` for image requirements.

## Usage

```bash
# Default: fullscreen mode
python3 main.py

# Different camera
python3 main.py --camera-index 1

# Windowed mode (for testing)
python3 main.py --windowed
```

### Controls

| Key | Action |
|-----|--------|
| SPACE | Next filter |
| B | Previous filter |
| 1-4 | Select filter directly |
| C | Capture screenshot |
| F | Toggle fullscreen |
| Q / ESC | Quit |

## Available Filters

| Filter | Anchor Point | Description |
|--------|-------------|-------------|
| Golden Crown | forehead | Crown sitting on top of head |
| Butterfly Wings | nose | Wings spread around the face |
| Fire Eyes | eyes_center | Fire effect over the eyes |
| Angel Halo | forehead | Halo floating above the head |

## Model Downloads

On first run, the application automatically downloads required models (~65MB total):
- `deploy.prototxt` - Face detector architecture
- `res10_300x300_ssd_iter_140000.caffemodel` - Face detector weights (~10MB)
- `lbfmodel.yaml` - Facial landmark model (~54MB)

Models are saved to `models/` directory.

## Troubleshooting

### Camera Not Found
```bash
# List available cameras
ls /dev/video*

# Try different indices
python3 main.py --camera-index 0
python3 main.py --camera-index 1
```

### OpenCV Import Error on ARM64
```bash
# Use system OpenCV instead of pip version
sudo apt install python3-opencv
pip3 uninstall opencv-python opencv-contrib-python
```

### Model Download Fails
```bash
# Download manually
mkdir -p models
cd models
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
wget https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
```

### Performance Tips
- Use 640x480 resolution (default)
- Close other applications
- Use adequate power supply

## Project Structure

```
Navesnak/
├── main.py             # Main application
├── face_detector.py    # OpenCV DNN face detection
├── filters.py          # Filter overlay logic
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── QUICKSTART.md       # Quick reference guide
├── models/             # Downloaded DNN models
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── lbfmodel.yaml
└── assets/
    └── filters/        # PNG filter images
```

## Technical Details

- **Face Detection**: OpenCV DNN SSD ResNet (Caffe model)
- **Landmarks**: OpenCV FacemarkLBF (68-point model)
- **Platform**: ARM64/aarch64 compatible (Raspberry Pi 4/5)
- **Python**: 3.11+ (tested with 3.13.5)
- **Performance**: Frame skipping (detection every 3rd frame), vectorized alpha blending

## License

This project is provided as-is for educational and entertainment purposes.
