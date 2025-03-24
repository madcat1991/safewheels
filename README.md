# SafeWheels

A Python application for processing video streams and files to detect vehicles and recognize license plates.

## Features

- Process RTSP camera streams or video files
- Vehicle detection using YOLOv8
- License plate detection and recognition using pre-trained YOLOv8 models
- Groups multiple images of the same vehicle to improve recognition accuracy
- Stores vehicle images and plate information for later review

## Requirements

- Python 3.13
- OpenCV
- PyTorch
- Ultralytics YOLOv8

## Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/safewheels.git
   cd safewheels
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

## Configuration

Edit the configuration file at `config/config.json`:

```json
{
  "rtsp_url": "rtsp://your-camera-ip:port/stream",
  "rtsp_username": "username",
  "rtsp_password": "password",
  "detection_interval": 1.0,
  "confidence_threshold": 0.5,
  "plate_confidence_threshold": 0.7,
  "grouping_time_window": 10,
  "storage_path": "data/vehicles",
  "max_stored_images": 1000
}
```

## Usage

### Process RTSP stream

Run the application using the RTSP URL defined in the config:

```
python -m safewheels.main
```

### Process video file

For debugging or offline analysis, you can process a video file:

```
python -m safewheels.main --video /path/to/your/video.mp4
```

## Data Storage

Detected vehicles and license plates are stored in the configured `storage_path`, including:
- Vehicle images
- License plate images (when detected)
- JSON records with timestamps, frame numbers, and detection data
