# SafeWheels

A Python application for processing video streams and files to detect vehicles and recognize license plates.

## Features

- Process RTSP camera streams or video files
- Vehicle detection using YOLOv8
- License plate detection using a [pre-trained](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8) YOLOv8 model
- License plate recognition using fast-plate-ocr with European plate models
- GPU acceleration for both detection and recognition (CUDA support)
- Fast processing with optimized preprocessing pipeline
- Groups multiple images of the same vehicle to improve recognition accuracy
- Stores vehicle images and plate information in SQLite database
- Real-time Telegram notifications with best vehicle images

## Requirements

- Python 3.13
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- fast-plate-ocr
- PyAV
- Python-Telegram-Bot

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

   This will install all necessary Python packages for vehicle detection, license plate recognition, and notifications.

## Configuration

Edit the configuration file at `config/config.json`:

```json
{
  "rtsp_url": "rtsp://your-camera-ip:port/stream",
  "rtsp_username": "username",
  "rtsp_password": "password",
  "vehicle_confidence_threshold": 0.4,
  "plate_detection_threshold": 0.3,
  "ocr_confidence_threshold": 0.2,
  "process_every_n_frames": 10,
  "storage_path": "data/vehicles",
  "vehicle_id_threshold_sec": 5,
  "db_filename": "detections.db",
  "images_dirname": "images",
  "check_interval_sec": 5,
  "telegram_token": "",
  "authorized_users": [
    123456789,
    987654321
  ],
  "timestamp_file": "last_processed.txt",
  "use_gpu": true,
  "batch_size": 4,
  "model_precision": "fp16",
  "cuda_device": 0
}
```

Key configuration parameters:
- `vehicle_confidence_threshold`: Minimum confidence for vehicle detection (0.0-1.0)
- `plate_detection_threshold`: Minimum confidence for license plate detection (0.0-1.0)
- `ocr_confidence_threshold`: Minimum confidence for OCR recognition (0.0-1.0)
- `process_every_n_frames`: Process every Nth frame from video streams for efficiency
- `vehicle_id_threshold_sec`: Time threshold to consider a vehicle as complete/unique
- `db_filename`: Name of the SQLite database file for storing detections
- `check_interval_sec`: How often to check for new completed vehicles (for notifications)
- `authorized_users`: Array of Telegram user IDs authorized to receive notifications
- `timestamp_file`: File to store the timestamp of last processed detection (for persistence across restarts)
- `use_gpu`: Whether to use GPU acceleration (true/false)
- `model_precision`: Model precision to use ('fp32' or 'fp16' for faster CUDA processing)
- `cuda_device`: CUDA device ID to use (0 for primary GPU)

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

### Use a specific configuration file

You can specify a custom configuration file path:

```
python -m safewheels.main --config /path/to/custom/config.json
```

## Data Storage

Detected vehicles and license plates are stored in the configured `storage_path`, including:
- Vehicle images in the specified `images_dirname` directory
- SQLite database (`db_filename`) with detection records and metadata

## Monitoring and Notifications

SafeWheels includes a monitoring script that periodically checks detection records and sends the best images to a Telegram chat.

### Setting up Telegram notifications

1. Create a Telegram bot using BotFather
   - Message @BotFather on Telegram
   - Use the `/newbot` command and follow the instructions
   - Save the API token you receive

2. Get the user IDs for authorized users
   - Have each authorized user send a message to the bot
   - Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Look for the `from` object in each message and note the `id` value for each user

3. Update your configuration in `config/config.json`:
   ```json
   {
     ...
     "db_filename": "detections.db",
     "check_interval_sec": 5,
     "telegram_token": "YOUR_BOT_TOKEN",
     "authorized_users": [
       123456789,  // First user's ID
       987654321   // Second user's ID
     ]
   }
   ```

### Running the monitor script

```bash
python scripts/monitor_and_notify.py -c /path/to/config.json
```

The script will identify completed vehicle detections and send the best image for each vehicle to all authorized Telegram users. For each vehicle, it selects the image with the highest confidence based on this priority:
1. Highest OCR confidence (plate recognized)
2. Highest plate detection confidence
3. Highest vehicle detection confidence

For more detailed information about the monitoring script, see the [scripts/README.md](scripts/README.md) file.
