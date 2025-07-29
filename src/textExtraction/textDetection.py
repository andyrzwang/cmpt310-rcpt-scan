import cv2
import numpy as np
import os
import sys
import pytesseract as pyt
from scipy.ndimage import interpolation as inter


from PIL import Image
import re
from datetime import datetime


# import preprocess modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocess')))
from file_full_path import folder_file_path
from grayscale import grayscale
from binarization import apply_binarization
from rescale import rescale
from fix_tilted_image import fix_tilted_image
import json

def image_preProcess(image):
    # Convert to grayscale
    # image = rescale(image)  # rescale
    gray = grayscale(image) # grayscale
    bin = apply_binarization(gray) #binarization
    # Fix skewed image
    upRight = fix_tilted_image(bin)  # fix skewed image
    return upRight


def findTotal(lines):
    # fakeTotal.json is a list of words that represents fake total text
    with open('src\\textExtraction\\fakeTotal.json', 'r') as f:
        fake_data = json.load(f)
    with open('src\\textExtraction\\trueTotal.json', 'r') as f:
        true_data = json.load(f)
    
    fake_total_list = fake_data.get('fakeTotalList', [])
    true_total_list = true_data.get('fakeTotalList', [])

    return 0.0



def findDate(lines):
    return None


def text_detection(image, original_image, image_file_Path):
    # pre process the format of the image ++++++++++
    if isinstance(image, tuple):
        # Pick the first valid NumPy array
        image = next((x for x in image if isinstance(x, np.ndarray)), None)
    

    text_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if len(image.shape) == 2:  # grayscale
        text_image = Image.fromarray(image)
    elif len(image.shape) == 3:  # BGR color
        text_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # store text_image to a file
    text_image.save('text_image.png')

    # pre process the format of the image ----------

    # text recognition, extraction and sort ++++++++++
    ocr_data = pyt.image_to_data(text_image, output_type = pyt.Output.DATAFRAME)
    

    # Clean the dataframe
    ocr_data = ocr_data.dropna(subset=['text'])
    ocr_data = ocr_data[ocr_data['text'].str.strip() != '']

    # Store OCR data to csv
    # ocr_data.to_csv('ocr_data.csv', index=False) # testing


    lines = (
        ocr_data.groupby(['block_num', 'line_num'])['text']
        .apply(lambda words: ' '.join(words.tolist()))
        .tolist()
    )
    # text recognition, extraction and sort ----------

    total = findTotal(lines)
    date = findDate(lines)
    

    # return total
    return total, date




if __name__ == "__main__":

    # Load the image
    file_name = '000.jpg'  # Replace with your image file name
    image_path = folder_file_path('images', file_name)
    image = cv2.imread(image_path)
    
    # pre-process the image
    pre_processed_image = image_preProcess(image)

    # Perform text detection+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    values = text_detection(
        pre_processed_image, # cv2 image
        pre_processed_image, # cv2 image 
        image_path, #path to the original image
        )
    # Perform text detection+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # get solution file

    imageSol = file_name.replace('.jpg', '.json')
    sol_path = folder_file_path('gdt', imageSol)
    with open(sol_path, 'r') as f:
        sol_data = json.load(f)
    total_value, date = sol_data.get('total')


    # check
    if values == total_value:
        print("The total value is correct.")
    else:
        print("The total value is incorrect.")
    
    if date == sol_data.get('date'):
        print("The date is correct.")
    else:
        print("The date is incorrect.")
    






