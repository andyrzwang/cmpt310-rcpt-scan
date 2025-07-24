# Importing Needed libraries
import sys
import os
import cv2
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_full_path import give_colored_Image_full_File_Path 

# Add the path to the parent folder of preprocess

# Importing the grayscale image conversion fucntion from grayscale.py
from grayscale import grayscale
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Test function to varify if an image has been successfully converted into grayscale 
def test_grayscale():

    # Testing the first image for simplicity 
    # path = 'cmpt310-rcpt-scan/data/dirty/large-receipt-image-dataset-SRD/1000-receipt.jpg'
    path = give_colored_Image_full_File_Path('1000-receipt.jpg')
    
    # Attaining the result of the image
    image = cv2.imread(path)
    result = grayscale(image)

    # Checking to see if the image is grayscale or not 
    assert result is not None, "Failed to load image"
    assert len(result.shape) == 2, "Image is not in grayscale"

    # Printing the result 
    print("✅ Test passed!")

if __name__ == "__main__":
    test_grayscale()
