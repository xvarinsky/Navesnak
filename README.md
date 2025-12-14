# Magic Mirror Face Filter

A fun, interactive Raspberry Pi application that uses a camera to show a live mirrored video with face detection and overlay filters (mustache, glasses, cat ears, unicorn horn, clown nose).

## Features

- **Live Video Mirror**: Real-time mirrored video feed from camera
- **Face Detection**: Automatic face detection using MediaPipe Face Mesh
- **Multiple Filters**: Switch between different fun filters
- **Fullscreen Display**: Optimized for HDMI display on Raspberry Pi
- **Keyboard Controls**: Easy filter switching and quit functionality
- **Transparent Overlays**: PNG filters with alpha blending

## Hardware Requirements

- Raspberry Pi 4 (or compatible)
- Camera: Raspberry Pi Camera Module **OR** USB webcam
- Display: External monitor connected via HDMI
- Raspberry Pi OS with Python 3

## Installation

### 1. Update System Packages

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install System Dependencies

```bash
sudo apt install -y python3-pip python3-opencv libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev libqtgui4 libqt4-test python3-pyqt5
```

**Note**: If you're using the Raspberry Pi Camera Module, make sure it's enabled:
```bash
sudo raspi-config
# Navigate to: Interface Options > Camera > Enable
# Reboot after enabling
```

### 3. Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

**Note**: On Raspberry Pi, you may need to use `pip3 install --user` if you encounter permission issues.

### 4. Add Filter Images

Place your filter PNG images in the `assets/filters/` directory:

- `mustache.png`
- `glasses.png`
- `cat_ears.png`
- `unicorn.png`
- `clown_nose.png`

See `assets/filters/README.md` for more details about filter image requirements.

## Usage

### Basic Usage

Run the application with default settings (camera index 0, fullscreen):

```bash
python3 main.py
```

### Command Line Options

```bash
# Use a different camera (e.g., USB webcam at index 1)
python3 main.py --camera-index 1

# Run in windowed mode (for testing on desktop)
python3 main.py --windowed

# Combine options
python3 main.py --camera-index 0 --windowed
```

### Controls

- **SPACE** - Switch to next filter
- **B** - Switch to previous filter
- **Q** or **ESC** - Quit application

## Camera Setup

### Raspberry Pi Camera Module

1. Enable the camera in `raspi-config`:
   ```bash
   sudo raspi-config
   ```
   Navigate to: Interface Options > Camera > Enable

2. Reboot:
   ```bash
   sudo reboot
   ```

3. Test the camera:
   ```bash
   libcamera-hello --list-cameras
   ```

**Note**: If using the Raspberry Pi Camera Module, you may need to use `libcamera-vid` or configure OpenCV to use it. For USB webcams, use `--camera-index 0` (or higher if multiple cameras).

### USB Webcam

Most USB webcams work out of the box. If you have multiple cameras, try different indices:

```bash
python3 main.py --camera-index 0  # First camera
python3 main.py --camera-index 1  # Second camera
```

## Troubleshooting

### Camera Not Found

- Check camera connection
- Try different camera indices: `--camera-index 0`, `--camera-index 1`
- For Raspberry Pi Camera: Ensure it's enabled in `raspi-config`
- Check if camera is in use by another application

### No Filters Displayed

- Ensure PNG files are in `assets/filters/` directory
- Check that filter images have proper transparency (alpha channel)
- Verify file names match exactly: `mustache.png`, `glasses.png`, etc.

### Performance Issues

- Reduce camera resolution in `main.py` (currently set to 640x480)
- Close other applications to free up resources
- Ensure adequate power supply for Raspberry Pi 4

### Fullscreen Issues

- Use `--windowed` flag for testing
- Check HDMI connection and display settings
- Try adjusting display resolution in Raspberry Pi settings

## Project Structure

```
Navesnak/
├── main.py                 # Main application entry point
├── filters.py              # Filter configuration and overlay logic
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── assets/
    └── filters/
        ├── README.md      # Filter image instructions
        ├── mustache.png   # (add your images here)
        ├── glasses.png
        ├── cat_ears.png
        ├── unicorn.png
        └── clown_nose.png
```

## Technical Details

- **Face Detection**: MediaPipe Face Mesh for accurate landmark detection
- **Image Processing**: OpenCV for video capture and image manipulation
- **Filter Overlay**: Alpha blending with automatic scaling and positioning
- **Performance**: Optimized for Raspberry Pi 4 with 640x480 resolution

## License

This project is provided as-is for educational and entertainment purposes.

## Credits

Built with:
- OpenCV (computer vision)
- MediaPipe (face detection and landmarks)
- Python 3

