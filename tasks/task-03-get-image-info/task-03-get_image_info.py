import numpy as np

def get_image_info(image):
    """
    Extracts metadata and statistical information from an image.

    Parameters:
    - image (numpy.ndarray): Input image.

    Returns:
    - dict: Dictionary containing image metadata and statistics.
    """
    
    ### START CODE HERE ###
    # take image dimensions 
    height, width = image.shape[:2]
    dtype = str(image.dtype)
    depth = 1 if len(image.shape) == 2 else image.shape[2]
    
    #min - max - mean - std
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    mean_val = float(np.mean(image))
    std_val = float(np.std(image))
    ### END CODE HERE ###

    return {
        "width": width,
        "height": height,
        "dtype": dtype,
        "depth": depth,
        "min_value": min_val,
        "max_value": max_val,
        "mean": mean_val,
        "std_dev": std_val
    }

# Example Usage:
sample_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
info = get_image_info(sample_image)

# Print results
for key, value in info.items():
    print(f"{key}: {value}")