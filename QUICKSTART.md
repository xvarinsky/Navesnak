# Magic Mirror Face Filter - Quick Start Guide

## Project Description

**Magic Mirror Face Filter** is an interactive Raspberry Pi application that:
- Captures live video from a camera (Raspberry Pi Camera or USB webcam)
- Displays a horizontally-mirrored video feed (like a real mirror)
- Detects faces using MediaPipe Face Mesh
- Overlays fun PNG filters (mustache, glasses, cat ears, unicorn horn, clown nose)
- Allows switching between filters using keyboard controls
- Runs fullscreen on HDMI display (or windowed for testing)

## Installation Commands (Raspberry Pi OS)

Run these commands in order:

```bash
# 1. Update system packages
sudo apt update
sudo apt upgrade -y

# 2. Install system dependencies
sudo apt install -y python3-pip python3-opencv libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test python3-pyqt5

# 3. Enable Raspberry Pi Camera (if using Pi Camera Module)
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
# Reboot after enabling: sudo reboot

# 4. Install Python dependencies
pip3 install -r requirements.txt
# Or if permission issues: pip3 install --user -r requirements.txt
```

## Requirements.txt

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

## How to Run

### Basic Usage (Fullscreen)
```bash
python3 main.py
```

### Use Different Camera
```bash
python3 main.py --camera-index 1
```

### Windowed Mode (for testing)
```bash
python3 main.py --windowed
```

### Controls
- **SPACE** - Next filter
- **B** - Previous filter  
- **Q** or **ESC** - Quit

## Before Running

1. **Add Filter Images**: Place PNG files in `assets/filters/`:
   - `mustache.png`
   - `glasses.png`
   - `cat_ears.png`
   - `unicorn.png`
   - `clown_nose.png`

2. **Connect Camera**: Ensure camera is connected and working

3. **Connect Display**: HDMI monitor should be connected for fullscreen mode

## Files Created

- `main.py` - Main application entry point
- `filters.py` - Filter configuration and overlay logic
- `requirements.txt` - Python dependencies
- `README.md` - Full documentation
- `assets/filters/` - Directory for filter PNG images

See `README.md` for detailed documentation and troubleshooting.

