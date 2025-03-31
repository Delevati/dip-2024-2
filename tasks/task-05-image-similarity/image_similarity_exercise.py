# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np
from PIL import Image # somente para load image
from pathlib import Path # evitar problemas com path em OS diferente

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    return {
        "mse": mse(i1, i2),
        "psnr": psnr(i1, i2),
        "ssim": ssim(i1, i2),
        "npcc": npcc(i1, i2)
    }

def mse(i1, i2):
    return np.mean((i1 - i2) ** 2)

def psnr(i1, i2):
    m = mse(i1, i2)
    if m == 0:
        return float('inf')
    return 10 * np.log10(1 / m)

def ssim(i1, i2):
    mu1, mu2 = np.mean(i1), np.mean(i2)
    sigma1, sigma2 = np.var(i1), np.var(i2)
    sigma12 = np.mean((i1 - mu1) * (i2 - mu2))
    c1, c2 = 0.01**2, 0.03**2
    return ((2*mu1*mu2 + c1) * (2*sigma12 + c2) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)))

def npcc(i1, i2):
    i1_mean, i2_mean = np.mean(i1), np.mean(i2)
    num = np.sum((i1 - i1_mean) * (i2 - i2_mean))
    den = np.sqrt(np.sum((i1 - i1_mean) ** 2) * np.sum((i2 - i2_mean) ** 2))
    return num / den

def main():
    script_dir = Path(__file__).parent

    i1_path = script_dir.parent.parent / "img" / "baboon.png"
    i2_path = script_dir.parent.parent / "img" / "lena.png"

    i1 = np.array(Image.open(i1_path).convert("L"), dtype=np.float32) / 255.0
    i2 = np.array(Image.open(i2_path).convert("L"), dtype=np.float32) / 255.0
    
    different_results = compare_images(i1, i2)
    print("\n Imagens diferentes(i1,i2):")
    print("MSE:", different_results["mse"])
    print("PSNR:", different_results["psnr"])
    print("SSIM:", different_results["ssim"])
    print("NPCC:", different_results["npcc"])

    same_results = compare_images(i1, i1)
    print("\n Mesmas imagens(i1,i1):")
    print("MSE:", same_results["mse"])
    print("PSNR:", same_results["psnr"])
    print("SSIM:", same_results["ssim"])
    print("NPCC:", same_results["npcc"])
    print("\n")

if __name__ == "__main__":
    main()