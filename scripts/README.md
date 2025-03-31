# SafeWheels Monitoring Scripts

This directory contains scripts for monitoring and notification capabilities of the SafeWheels system.

## Monitor and Notify Script

The `monitor_and_notify.py` script provides real-time monitoring of vehicle detections and sends notifications to Telegram with the best image of each detected vehicle.

### Prerequisites

- A properly configured SafeWheels system with vehicle detection running
- A Telegram bot token
- A Telegram chat ID where notifications will be sent

### Configuration

The script uses the same configuration file as the main SafeWheels application, but specifically requires the following settings to be set:

```json
{
  "storage_path": "data/vehicles",
  "csv_filename": "detections.csv",
  "images_dirname": "images",
  "vehicle_id_threshold_sec": 5,
  "telegram_token": "YOUR_BOT_TOKEN",
  "telegram_chat_id": "YOUR_CHAT_ID"
}
```

Configuration parameters:

| Parameter | Description | Required |
|-----------|-------------|----------|
| `storage_path` | Directory where vehicle detection data is stored | Yes |
| `csv_filename` | Name of the CSV file containing detection records | Yes |
| `images_dirname` | Name of the directory containing detection images | Yes |
| `vehicle_id_threshold_sec` | Time threshold in seconds to consider a vehicle detection sequence as complete | Yes |
| `telegram_token` | Your Telegram bot API token | Yes |
| `telegram_chat_id` | The chat ID where notifications should be sent | Yes |

### Usage

Run the script with:

```bash
python monitor_and_notify.py -c /path/to/config.json
```

Command-line arguments:

- `-c, --config`: Path to the configuration file (default: config/config.json)

### How It Works

1. The script continuously monitors the CSV file containing vehicle detection records
2. When a vehicle is detected, it tracks all images captured for that vehicle
3. After a configurable time threshold (default: 5 seconds) with no new detections for that vehicle, it considers the detection sequence complete
4. It then selects the best image based on confidence scores:
   - First priority: Image with highest OCR confidence (plate recognized)
   - Second priority: Image with highest plate detection confidence
   - Last priority: Image with highest vehicle detection confidence
5. The selected image is sent to the configured Telegram chat with relevant details

### Continuous Operation

The script is designed to run continuously. To ensure it stays running even after system reboots, consider using a service manager like systemd (Linux) or creating a launch agent (macOS).

Example systemd service (Linux):

```
[Unit]
Description=SafeWheels Monitor and Notify Service
After=network.target

[Service]
User=yourusername
WorkingDirectory=/path/to/safewheels
ExecStart=/usr/bin/python /path/to/safewheels/scripts/monitor_and_notify.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Troubleshooting

If you're not receiving notifications:

1. Check that the Telegram bot token and chat ID are correctly configured
2. Ensure the bot has permission to send messages to the specified chat
3. Verify that the vehicle detection system is correctly writing to the CSV file
4. Check the console output or logs for any error messages