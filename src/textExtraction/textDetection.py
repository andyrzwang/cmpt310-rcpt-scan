import cv2
import numpy as np
import os
import sys

# import preprocess modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocess')))
from file_full_path import folder_file_path
from grayscale import grayscale
from binarization import apply_binarization
from rescale import rescale
import json

def image_preProcess(image):
    # Convert to grayscale
    scale = rescale(image)  # rescale
    gray = grayscale(scale) # grayscale
    bin = apply_binarization(gray) #binarization
    return bin

def text_detection(image, original_image):
    print('hello world')





if __name__ == "__main__":
    # Example usage
    file_name = '000.jpg'  # Replace with your image file name
    image_path = folder_file_path('images', file_name)

    
    # Load the image
    image = cv2.imread(image_path)

    pre_processed_image = image_preProcess(image)
    
    values = text_detection(pre_processed_image, pre_processed_image)

    print(values)

    # imageSol = '000.json'
    # sol_path = folder_file_path('gdt', imageSol)
    # with open(sol_path, 'r') as f:
    #     sol_data = json.load(f)
    # total_value = sol_data.get('total')
    






