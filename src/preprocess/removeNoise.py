import cv2
import numpy as np

import os
import sys

# Importing the grayscale image conversion function from grayscale.py
# use the following for testing only
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from grayscale import grayscale
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from file_full_path import give_colored_Image_full_File_Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from binarization import apply_binarization 


def removeNoise(image):
    # cv2.morphologyEx


    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations = 1)

    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

    return opening



# test function to verify if an image has been successfully converted into grayscale
# must comment out the test function (main)

if __name__ == "__main__":
    # Example usage
    image_path = give_colored_Image_full_File_Path('1008-receipt.jpg')

    print(f"Input image path: {image_path}")  # Debug print

    # Read the image
    image_file = cv2.imread(image_path)


    grayscale_image = grayscale(image_file)


    # call function
    binarized_image = apply_binarization(grayscale_image)

    # call function to remove noise ++++++++ Test function in this file
    noiseless_image = removeNoise(binarized_image)

    # store image

    filesave = noiseless_image
    output_filename = '1000-receipt-noiseless.jpg'
    script_dir = os.path.dirname(os.path.abspath(__file__)) # current file's dir
    output_path = os.path.join(script_dir, output_filename)

    cv2.imwrite(output_path, filesave)

