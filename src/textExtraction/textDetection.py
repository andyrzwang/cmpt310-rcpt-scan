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
    
    gray = grayscale(image) # grayscale
    bin = apply_binarization(gray) #binarization
    # Fix skewed image
    upRight = fix_tilted_image(bin)  # fix skewed image
    # scaled = rescale(upRight)  # rescale ----- do not use
    return upRight


def findTotal(lines, ocr_data):
    '''
    lines = clean and sorted of ocr_data
    '''
    # fakeTotal.json is a list of words that represents fake total text
    with open('src\\textExtraction\\fakeTotal.json', 'r') as f:
        fake_data = json.load(f)
    with open('src\\textExtraction\\trueTotal.json', 'r') as f:
        true_data = json.load(f)
    
    fake_total_list = fake_data.get('fakeTotalList', [])
    true_total_list = true_data.get('trueTotalList', [])

    for i in true_total_list:
        i = i.upper()
        # print(i)

    GuessedTotal = []

    for i in range(len(lines)):
        line = lines[i]
        upper_line = line.upper()        
        # Check if the line contains any of the true total keywords
        if any(true_kw in upper_line for true_kw in true_total_list):
            # check if the line contains any of the fake total keywords
            if not any(fake_kw in upper_line for fake_kw in fake_total_list):
                
                # extrac the total value from the line
                matches = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d{2})|\$\d+(?:\.\d{2})?', line)
                if matches:
                    cleaned = re.sub(r'[^\d\.]', '', matches[0])  # strip $, commas, etc.
                    total_value = float(cleaned)
                    GuessedTotal.append(total_value)
            
           
    # print(max(GuessedTotal) if GuessedTotal else None)
    return max(GuessedTotal) if GuessedTotal else None
    



def findDate(lines, ocr_data):
    '''
    lines = clean and sorted of ocr_data
    '''
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
    # text_image.save('text_image.png')

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

    total = findTotal(lines, ocr_data)
    date = findDate(lines, ocr_data)
    

    # return total
    return total, date













# test functions
if __name__ == "__main__":

    # # user input file number
    # file_number = input("Enter the file number (e.g., 006): ")
    # file_number = file_number.strip()  # Remove any leading/trailing whitespace
    # # convert to 3-digit format
    # file_number = file_number.zfill(3)  # Ensure it's 3 digits
    # # print(f"File number: {file_number}")
    # file_name = f"{file_number}.jpg"  # Construct the file name

    # Load the image
    file_name = '015.jpg' 
    image_path = folder_file_path('images', file_name)
    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist.")
        sys.exit(1)
    image = cv2.imread(image_path)
    
    # pre-process the image
    pre_processed_image = image_preProcess(image)

    # Perform text detection+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    total, date = text_detection(
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
    sol_total = sol_data.get('total')
    # conver sol_total to float
    sol_total = re.sub(r'[^\d.]', '', str(sol_total))  # remove any non-numeric characters
    sol_total = float(sol_total) if sol_total else 0.0  # convert
    sol_date = sol_data.get('date')


    # compare the results
    
    print(f"Total: {total}, Date: {date}")
    print(f"Solution Total: {sol_total}, Solution Date: {sol_date}")
    # check
    if float(total) == float(sol_total):
        print("The total value is correct.")
    else:
        print("The total value is incorrect.")
    
    # if date == sol_date:
    #     print("The date is correct.")
    # else:
    #     print("The date is incorrect.")
    






