import sys
import os
import cv2
import numpy as np

# Add the path to the parent folder of preprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing the grayscale image conversion function from grayscale.py
from grayscale import grayscale

# Test function to verify if an image has been successfully converted into grayscale 
def test_grayscale():
    
    # file directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # project file global
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))

    # print(script_dir) # debug print
    
    project_root = project_root[0:-4:1]
    print(project_root)
    # Path to the input image relative to project root

    filename = '1000-receipt.jpg'
    totalPath = project_root + '\data\dirty\large-receipt-image-dataset-SRD' + "\\" + filename
    input_path = os.path.join(project_root, totalPath)
    # print(f"Resolved input path: {input_path}")  # Debug print

    # Read image directly in grayscale mode
    # result = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE) # Debug print



    result = grayscale(input_path)

    # Check if the result is valid
    assert result is not None, f"Failed to load or process image at: {input_path}"
    assert len(result.shape) == 2, "Image is not in grayscale"

    # Save the grayscale image to the same folder as this Python script
    output_filename = '1000-receipt-gray.jpg'
    output_path = os.path.join(script_dir, output_filename)

    cv2.imwrite(output_path, result)
    # print("done") # debug print



if __name__ == "__main__":
    test_grayscale()
