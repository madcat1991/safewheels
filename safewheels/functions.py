import cv2


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
