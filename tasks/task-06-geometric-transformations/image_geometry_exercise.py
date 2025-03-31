# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Usei somente para carregar a imagem, não para transformações
from pathlib import Path # Garantir que encontre a imagem em diferentes OS

def translate_image(img, dx=30, dy=30):
    h, w = img.shape
    result = np.zeros_like(img)

    y_src, y_dst = 0, dy
    y_end_src, y_end_dst = h - dy, h

    x_src, x_dst = 0, dx
    x_end_src, x_end_dst = w - dx, w

    result[y_dst:y_end_dst, x_dst:x_end_dst] = img[y_src:y_end_src, x_src:x_end_src]
    
    return result

def rotate_image(img):
    return np.flip(img.T, axis=0)

def stretch_image_90(img, scale = 1.5):
    h, w = img.shape
    new_w = int(w * scale)
    result = np.zeros((h, new_w))

    for y in range(h):
        for x in range(new_w):
            src_x = x / scale

            x0 = int(src_x)
            if x0 >= w - 1:
                result[y, x] = img[y, w-1]
            else:
                dx = src_x - x0
                result[y, x] = (1 - dx) * img[y, x0] + dx * img[y, x0 + 1]

    return result

def mirror_image(img):
    return img[:, ::-1]
            
def barrel_distort_image(img, k=0.2):
    h, w = img.shape
    result = np.zeros_like(img)
    
    center_y, center_x = h / 2, w / 2
    
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    
    y_norm = (y_coords - center_y) / center_y
    x_norm = (x_coords - center_x) / center_x
    
    r_squared = x_norm**2 + y_norm**2
    
    factor = 1 + k * r_squared
    
    y_distorted = y_norm * factor
    x_distorted = x_norm * factor
    
    y_distorted = y_distorted * center_y + center_y
    x_distorted = x_distorted * center_x + center_x
    
    valid_mask = (
        (y_distorted >= 0) & 
        (y_distorted < h - 1) & 
        (x_distorted >= 0) & 
        (x_distorted < w - 1)
    )
    
    dest_y, dest_x = y_coords[valid_mask], x_coords[valid_mask]
    src_y, src_x = y_distorted[valid_mask], x_distorted[valid_mask]
    
    src_y0, src_x0 = src_y.astype(int), src_x.astype(int)
    src_y1, src_x1 = src_y0 + 1, src_x0 + 1
    
    wy1, wx1 = src_y - src_y0, src_x - src_x0
    wy0, wx0 = 1 - wy1, 1 - wx1
    
    v00 = img[src_y0, src_x0]
    v01 = img[src_y0, src_x1]
    v10 = img[src_y1, src_x0]
    v11 = img[src_y1, src_x1]
    
    result[dest_y, dest_x] = (
        wy0 * (wx0 * v00 + wx1 * v01) +
        wy1 * (wx0 * v10 + wx1 * v11)
    )
    
    return result

def apply_geometric_transformations(img: np.ndarray) -> dict:
    return {
        "translated": translate_image(img),
        "rotated": rotate_image(img),
        "stretched": stretch_image_90(img),
        "mirrored": mirror_image(img),
        "distorted": barrel_distort_image(img)
    }

def main():
    script_dir = Path(__file__).parent

    img_path = script_dir.parent.parent / "img" / "baboon.png"

    pil_img = Image.open(img_path).convert('L')
    img = np.array(pil_img)
    
    results = apply_geometric_transformations(img)
    
    Image.fromarray(img).show(title="Original")
    Image.fromarray(results["translated"]).show(title="Translated")
    Image.fromarray(results["rotated"]).show(title="Rotated")
    Image.fromarray(results["stretched"]).show(title="Stretched")
    Image.fromarray(results["mirrored"]).show(title="Mirrored")
    Image.fromarray(results["distorted"]).show(title="Barrel Distorted")

if __name__ == "__main__":
    main()