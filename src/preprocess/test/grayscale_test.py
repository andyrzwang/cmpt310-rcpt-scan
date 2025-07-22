# Importing Needed libraries
import sys
import os
import cv2
import numpy as np

# Add the path to the parent folder of preprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing the grayscale image conversion fucntion from grayscale.py
from grayscale import grayscale

# Test function to varify if an image has been successfully converted into grayscale 
def test_grayscale():

    # Testing the first image for simplicity 
    path = 'cmpt310-rcpt-scan/data/dirty/large-receipt-image-dataset-SRD/1000-receipt.jpg'
    
    # Attaining the result of the image
    result = grayscale(path)

    # Checking to see if the image is grayscale or not 
    assert result is not None, "Failed to load image"
    assert len(result.shape) == 2, "Image is not in grayscale"

    # Printing the result 
    print("✅ Test passed!")

if __name__ == "__main__":
    test_grayscale()
