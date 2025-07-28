import cv2
import numpy as np
import os
import sys
import pytesseract

from PIL import Image
import re
from datetime import datetime


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

def text_detection(image, original_image, image_file_Path):
    # fakeTotal.json is a list of words that represents fake total text
    with open('src\\textExtraction\\fakeTotal.json', 'r') as f:
        fake_data = json.load(f)
    with open('src\\textExtraction\\trueTotal.json', 'r') as f:
        true_data = json.load(f)
    
    fake_total_list = fake_data.get('fakeTotalList', [])
    true_total_list = true_data.get('fakeTotalList', [])




    # Test pytesseract version
    pytesseract.pytesseract.tesseract_cmd = r"c:\\users\\wangr\\appdata\\local\\programs\\python\\python310\\lib\\site-packages\\Tesseract-OCR\\tesseract.exe"

    print("pytesseract version:", pytesseract.get_tesseract_version())

    # extract text using pytesseract
    # text_image = Image.open(image_path)
    # text = pyt.image_to_string(text_image)
    # print(text)


    # extract text using pytesseract


    return 1










if __name__ == "__main__":
    # Example usage
    file_name = '000.jpg'  # Replace with your image file name
    image_path = folder_file_path('images', file_name)

    
    # Load the image
    image = cv2.imread(image_path)
    pre_processed_image = image_preProcess(image)



    
    
    values = text_detection(
        pre_processed_image, pre_processed_image, # image and original
        image_path, # This is the path to the original image
        )

    print(values)

    # imageSol = '000.json'
    # sol_path = folder_file_path('gdt', imageSol)
    # with open(sol_path, 'r') as f:
    #     sol_data = json.load(f)
    # total_value = sol_data.get('total')
    






