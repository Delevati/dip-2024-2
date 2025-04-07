# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 # Somente para carregar as imagens
import numpy as np
from skimage import exposure # Preferi fazer manualmente, mas poderia ter utilizado para fazer os histograms

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    
    matched_img = np.zeros_like(source_img)

    # Loop pra acessar e processar todas as linhas/colunas/canais
    for c in range(3):
        src_channel = source_img[:, :, c]
        ref_channel = reference_img[:, :, c]
        
        # Calcula os histogramas
        src_hist, _ = np.histogram(src_channel.flatten(), 256, [0, 256], density=True)
        ref_hist, _ = np.histogram(ref_channel.flatten(), 256, [0, 256], density=True)
        
        # Calcula as distribuições (CDF) (histogramas cumulativos)
        src_cdf = src_hist.cumsum()
        ref_cdf = ref_hist.cumsum()
        
        # Normaliza distribuição (CDF) entre [0, 1]
        src_cdf_normalized = src_cdf / src_cdf[-1]
        ref_cdf_normalized = ref_cdf / ref_cdf[-1]
        
        # Start mapping (lookup table)
        interp_values = np.interp(src_cdf_normalized, ref_cdf_normalized, np.arange(256))
        
        # Apply mapping
        matched_channel = interp_values[src_channel].astype(np.uint8)
        matched_img[:, :, c] = matched_channel

    return matched_img

def print_ppm_p3(image, max_rows=10, max_cols=10):
    height, width, channels = image.shape

    sample_height = min(height, max_rows)
    sample_width = min(width, max_cols)
    
    max_value = int(np.max(image))
    
    # Print PPM header 
    # (formato texto onde cada pixel é 3 valores R G B seguidos)
    print(f"P3")
    print(f"{sample_width} {sample_height} {channels}")
    print(f"{max_value}")
    
    for y in range(sample_height):
        for x in range(sample_width):
            r = int(image[y, x, 0])
            g = int(image[y, x, 1])
            b = int(image[y, x, 2])
            print(f"{r} {g} {b} ", end="")
        print()
    
if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    
    img_A = cv2.imread('source.jpg')
    img_B = cv2.imread('reference.jpg')
    
    img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
    img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
    
    img_A_transformed = match_histograms_rgb(img_A, img_B)
    
    print_ppm_p3(img_A_transformed)