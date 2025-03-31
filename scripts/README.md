# SafeWheels Monitoring Scripts

This directory contains scripts for monitoring and notification capabilities of the SafeWheels system.

## Monitor and Notify Script

The `monitor_and_notify.py` script provides real-time monitoring of vehicle detections and sends notifications to Telegram with the best image of each detected vehicle.

### Prerequisites

- A properly configured SafeWheels system with vehicle detection running
- A Telegram bot token
- A Telegram chat ID where notifications will be sent

### Configuration

The script uses the same configuration file as the main SafeWheels application. See the main [README.md](../README.md) for complete configuration options.

Important configuration parameters for this script:

| Parameter | Description |
|-----------|-------------|
| `vehicle_id_threshold_sec` | Time threshold in seconds to consider a vehicle detection sequence as complete |
| `check_interval_sec` | How often (in seconds) the script checks for new completed vehicles |
| `telegram_token` | Your Telegram bot API token |
| `telegram_chat_id` | The chat ID where notifications should be sent |

### Usage

Run the script with:

```bash
python monitor_and_notify.py -c /path/to/config.json
```

Command-line arguments:

- `-c, --config`: Path to the configuration file (required)

### How It Works

1. The script periodically checks the database at the interval specified by `check_interval_sec`
2. It identifies vehicles that haven't been detected for at least `vehicle_id_threshold_sec` seconds (considered "completed")
3. For each completed vehicle not already processed, it selects the best image based on confidence scores using database ranking
4. The selected image is sent to the configured Telegram chat with relevant details
5. The script tracks the last processed timestamp to avoid sending duplicate notifications

### Efficient Implementation

The script uses an optimized single SQL query with window functions to:

- Find vehicles detected after the last processing timestamp
- Filter for vehicles that haven't been detected for the threshold period
- Rank images by confidence (OCR > plate detection > vehicle detection)
- Select only the best image for each vehicle

This database-driven approach is efficient even with large numbers of detections and minimizes memory usage.

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
ExecStart=/usr/bin/python /path/to/safewheels/scripts/monitor_and_notify.py -c /path/to/config.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Troubleshooting

If you're not receiving notifications:

1. Check the Telegram configuration
   - Verify the bot token and chat ID in config.json
   - Ensure the bot has permission to send messages to the specified chat
   - Test the bot by sending it a direct message

2. Check the database setup
   - Ensure the SQLite database file exists at the configured location
   - Verify that the database contains records (use `sqlite3 path/to/database.db` and run `.tables` and `SELECT COUNT(*) FROM detections;`)
   - Check that the database has the correct schema (run `.schema detections` in SQLite)
   
3. Review logs and permissions
   - Check the console output for error messages
   - Verify the script has read access to the database and images directory
   - Ensure the images referenced in the database exist in the images directory