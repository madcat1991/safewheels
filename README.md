# SafeWheels

A Python application for processing RTSP camera streams to detect vehicles and recognize license plates.

<p align="center">
   <img width="326" alt="Car plate recognition front example" src="https://github.com/user-attachments/assets/cf33b017-b017-43cb-ac99-86a85cca560a" />
   <img width="326" alt="Car plate recognition rear example" src="https://github.com/user-attachments/assets/c01db2c7-1820-466b-8e50-d336f8a86925" />
</p>


## Features

- Process RTSP camera streams
- Vehicle detection using YOLOv8
- License plate detection using a [pre-trained](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8) YOLOv8 model
- License plate recognition using fast-plate-ocr with European plate models
- GPU acceleration for both detection and recognition (CUDA support)
- Fast processing with optimized preprocessing pipeline
- Groups multiple images of the same vehicle to improve recognition accuracy
- Stores vehicle images and plate information in SQLite database
- Near real-time Telegram notifications with best vehicle images
- Automatic cleanup of processed records to manage disk space

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
  "model_precision": "fp16",
  "cuda_device": 0
}
```

Key configuration parameters:
- `rtsp_url`: URL of your RTSP camera stream
- `rtsp_username` and `rtsp_password`: Credentials for your RTSP stream (if required)
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
python -m safewheels.scripts.watch_and_detect --config config/config.json
```

### Use a specific configuration file

You can specify a custom configuration file path:

```
python -m safewheels.scripts.watch_and_detect --config /path/to/custom/config.json
```

## Data Storage

Detected vehicles and license plates are stored in the configured `storage_path`, including:
- Vehicle images in the specified `images_dirname` directory
- SQLite database (`db_filename`) with detection records and metadata

## Scripts

SafeWheels includes several utility scripts to help with monitoring, recognition, and system maintenance.

### Watch and Detect

The `watch_and_detect.py` script continuously monitors a video stream for vehicle detections:

```bash
python -m safewheels.scripts.watch_and_detect --config config/config.json
```

### Recognize and Notify

The `recognize_and_notify.py` script processes detected vehicles, recognizes license plates, and sends notifications:

```bash
python -m safewheels.scripts.recognize_and_notify --config config/config.json
```

The script periodically checks the database at the interval specified by `check_interval_sec`:
1. It identifies vehicles that haven't been detected for at least `vehicle_id_threshold_sec` seconds (considered "completed")
2. For each completed vehicle not already processed, it selects the best image based on confidence scores
3. It recognizes license plates from the detected plate areas
4. The selected image is sent to all authorized Telegram users with relevant details
5. The script tracks the last processed timestamp and saves it to the file specified by `timestamp_file` to avoid duplicate notifications

### Cleanup Processed

The `cleanup_processed.py` script helps manage disk space by removing detections that don't have corresponding recognition records:

```bash
python -m safewheels.scripts.cleanup_processed --config config/config.json --min-age 24
```

Command-line arguments:
- `--min-age`: Minimum age in hours for records to be eligible for cleanup (default: 24)
- `--dry-run`: Run in simulation mode without actually deleting files

This script finds and removes:
1. Detection records in the database that don't have a corresponding recognition record
2. The associated image files for those detections
3. Only processes records older than the specified minimum age

## Setting up Telegram notifications

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

## Continuous Operation

These scripts are designed to run continuously. To ensure they stay running even after system reboots, consider using a service manager like systemd (Linux) or creating a launch agent (macOS).

Example systemd service (Linux):

```
[Unit]
Description=SafeWheels Vehicle Detection Service
After=network.target

[Service]
User=yourusername
WorkingDirectory=/path/to/safewheels
ExecStart=/usr/bin/python -m safewheels.scripts.watch_and_detect --config /path/to/config.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Troubleshooting

If you're not receiving notifications:

1. Check the Telegram configuration
   - Verify the bot token in config.json
   - Ensure your Telegram user ID is correctly listed in the `authorized_users` array
   - Verify that each user ID is a number, not a string (e.g., `123456789`, not `"123456789"`)
   - Confirm that each authorized user has started a chat with the bot
   - Test the bot by sending it a direct message

2. Check the database setup
   - Ensure the SQLite database file exists at the configured location
   - Verify that the database contains records (use `sqlite3 path/to/database.db` and run `.tables` and `SELECT COUNT(*) FROM detections;`)
   - Check that the database has the correct schema (run `.schema detections` in SQLite)

3. Review logs and permissions
   - Check the console output for error messages
   - Verify the script has read access to the database and images directory
   - Ensure the images referenced in the database exist in the images directory
