import cv2


def draw_detection_on_image(img, plate_bbox, plate_number=None):
    """
    Draw bounding box and plate number on an image.

    Args:
        img: The original image
        plate_bbox: Bounding box coordinates for the license plate [x, y, w, h]
        plate_number: Recognized license plate number (optional)

    Returns:
        Image with bounding box and text drawn on it
    """
    # Create a copy of the image to avoid modifying the original
    result_img = img.copy()

    # Draw bounding box if available
    if plate_bbox is not None:
        px, py, pw, ph = plate_bbox
        # Draw the bounding box in green
        cv2.rectangle(result_img, (px, py), (px + pw, py + ph), (0, 153, 255), 2)

        # Add plate number if available
        if plate_number is not None:
            # Background for text
            text = f"{plate_number}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(result_img, (px, py - text_height - 10), (px + text_width + 10, py), (0, 0, 255), -1)
            # Text in white
            cv2.putText(result_img, text, (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return result_img


def get_compute_params(config):
    """
    Get compute parameters from the configuration.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        tuple: A tuple containing use_gpu, device, and model_precision.
    """
    use_gpu = config.get('use_gpu', True)
    device_type = config.get('device_type', None)  # Auto-detect if None
    model_precision = config.get('model_precision', 'fp16')
    cuda_device = config.get('cuda_device', 0)

    # If CUDA device is specified, update device name
    device = None
    if device_type == 'cuda' and cuda_device is not None:
        device = f'cuda:{cuda_device}'
    else:
        device = device_type

    return use_gpu, device, model_precision


def add_padding_to_bbox(bbox, image_shape, padding_percent=0.1, min_padding=5):
    """
    Add padding around a bounding box to include more context.

    Args:
        bbox: Original bounding box (x, y, w, h)
        image_shape: Shape of the image (height, width, ...)
        padding_percent: Percentage of bbox dimensions to add as padding
        min_padding: Minimum padding in pixels

    Returns:
        Padded bounding box (x, y, w, h)
    """
    x, y, w, h = bbox
    height, width = image_shape[:2]

    # Calculate padding (percentage of dimensions)
    pad_x = max(int(w * padding_percent), min_padding)
    pad_y = max(int(h * padding_percent), min_padding)

    # Create padded coordinates (ensuring they stay within image boundaries)
    padded_x = max(0, x - pad_x)
    padded_y = max(0, y - pad_y)
    padded_w = min(width - padded_x, w + 2 * pad_x)
    padded_h = min(height - padded_y, h + 2 * pad_y)

    return (padded_x, padded_y, padded_w, padded_h)


def to_grayscale(img):
    """
    Convert an image to grayscale.

    Args:
        img: Input image

    Returns:
        Grayscale image
    """
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray_plate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_plate = img

    return gray_plate
