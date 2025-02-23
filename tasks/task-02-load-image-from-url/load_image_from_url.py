import argparse
import numpy as np
import cv2 as cv
# Tive que adicionar request, observei que ela está inclusa no requirements, tentei algumas alternativas como o 
# cv.videoCapture, e o cv.imdecode diretamente mas sem êxito, então resolvi adicionar a request para finalizar o trabalho.
import requests

def load_image_from_url(url=None, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###
    # python3 load_image_from_url.py "https://tm.ibxk.com.br/2012/3/materias/529462367269129.jpg"
    if url is None:
        # Argparse to get url
        parser = argparse.ArgumentParser(description="Load an image from a URL")
        parser.add_argument('url', type=str, help="URL of the image")
        args = parser.parse_args()

        url = args.url

        # Define grayscale flag
        flags = cv.IMREAD_GRAYSCALE

    # Getting image from URL
    response = requests.get(url)
    
    # Convert content to array uint 8
    image_array = np.frombuffer(response.content, dtype=np.uint8)

    # Decode image with cv2.imdecode
    image = cv.imdecode(image_array, **kwargs, flags=flags)
    
    if image is None:
        raise ValueError("Failed: empty image")
    
    print("Array:")
    print(image)
    ### END CODE HERE ###
    
    return image

load_image_from_url()
# load_image_from_url()
