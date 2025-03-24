"""
Configuration utilities for SafeWheels application.
"""
import os
import json
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.json')


def load_config(config_path=None):
    """
    Load application configuration from a JSON file.

    Args:
        config_path (str, optional): Path to config file. Defaults to config/config.json.

    Returns:
        dict: Configuration dictionary.
    """
    if not config_path:
        config_path = DEFAULT_CONFIG_PATH

    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}. Creating default config.")
        create_default_config(config_path)

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return create_default_config(config_path)


def create_default_config(config_path):
    """
    Create a default configuration file.

    Args:
        config_path (str): Path to save the default config.

    Returns:
        dict: Default configuration dictionary.
    """
    default_config = {
        "rtsp_url": "",
        "rtsp_username": "",
        "rtsp_password": "",
        "detection_interval": 1.0,  # Seconds between processing frames
        "confidence_threshold": 0.5,  # Minimum confidence for vehicle detection
        "plate_confidence_threshold": 0.7,  # Minimum confidence for plate recognition
        "grouping_time_window": 10,  # Seconds to group detections of the same vehicle
        "storage_path": "data/vehicles",  # Where to store detected vehicle images
        "max_stored_images": 1000  # Maximum number of images to keep
    }

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    try:
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        logger.info(f"Default config created at {config_path}")
        return default_config
    except Exception as e:
        logger.error(f"Error creating default config: {e}")
        return default_config
