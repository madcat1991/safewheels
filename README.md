# SafeWheels

A Python application for processing video streams and files to detect vehicles and recognize license plates.

## Features

- Process RTSP camera streams or video files
- Vehicle detection using YOLOv8
- License plate detection using a [pre-trained](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8) YOLOv8 model
- License plate recognition using Tesseract OCR with multi-language support (English, German)
- Fast processing with optimized preprocessing pipeline
- Groups multiple images of the same vehicle to improve recognition accuracy
- Stores vehicle images and plate information for later review

## Requirements

- Python 3.13
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- Tesseract OCR 5.x
- PyAV

### Tesseract OCR Installation

Tesseract OCR is required and must be installed separately:

#### macOS:
```
brew install tesseract
brew install tesseract-lang  # for language support
```

#### Ubuntu/Debian:
```
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-eng tesseract-ocr-deu  # for English and German
```

#### Windows:
1. Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add the Tesseract installation directory to your PATH

## Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/safewheels.git
   cd safewheels
   ```

2. Install Python dependencies
   ```
   pip install -r requirements.txt
   ```

   This will install all necessary Python packages including pytesseract (Python wrapper for Tesseract OCR).

## Configuration

Edit the configuration file at `config/config.json`:

```json
{
  "rtsp_url": "rtsp://your-camera-ip:port/stream",
  "rtsp_username": "username",
  "rtsp_password": "password",
  "detection_interval": 1.0,
  "confidence_threshold": 0.4,
  "plate_confidence_threshold": 0.25,
  "grouping_time_window": 10,
  "storage_path": "data/vehicles",
  "max_stored_images": 1000
}
```

Key configuration parameters:
- `confidence_threshold`: Minimum confidence for vehicle detection (0.0-1.0)
- `plate_confidence_threshold`: Minimum confidence for license plate recognition (0.0-1.0)
  - Lower values will detect more plates but may introduce false positives
  - Higher values will be more accurate but may miss some plates

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
- CSV file with all detection records

## Monitoring and Notifications

SafeWheels includes a monitoring script that can continuously check detection records and send the best images to a Telegram chat.

### Setting up Telegram notifications

1. Create a Telegram bot using BotFather
   - Message @BotFather on Telegram
   - Use the `/newbot` command and follow the instructions
   - Save the API token you receive

2. Get your chat ID
   - Add the bot to a group or start a chat with it
   - Send a message to the bot
   - Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Look for the `chat` object and note the `id` value

3. Update your configuration in `config/config.json`:
   ```json
   {
     ...
     "telegram_token": "YOUR_BOT_TOKEN",
     "telegram_chat_id": "YOUR_CHAT_ID"
   }
   ```

### Running the monitor script

```bash
python scripts/monitor_and_notify.py -c /path/to/config.json
```

Arguments:
- `-c, --config`: Path to the configuration file (default: config/config.json)

The script will monitor for completed vehicle detections and send images with the highest confidence to your Telegram chat.

Required configuration settings for the monitoring script:
- `storage_path`: Location of your detection data
- `csv_filename`: Name of the CSV file with detections
- `images_dirname`: Folder containing the images
- `telegram_token`: Your Telegram bot token
- `telegram_chat_id`: Your Telegram chat ID
- `vehicle_id_threshold_sec`: Time threshold to consider a vehicle detection as complete
