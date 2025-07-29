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
from textDetection import text_detection, image_preProcess
import json

found_total_list = []
found_date_list = []

sol_total_list = []
sol_date_list = []


for i in range(1, 100):
    i = str(i)
    file_number = i.zfill(3)
    file_name = f"{file_number}.jpg"
    image_path = folder_file_path('images', file_name)

    image = cv2.imread(image_path)
    pre_processed_image = image_preProcess(image)

    total, date = text_detection(
        pre_processed_image, # cv2 image
        pre_processed_image, # cv2 image 
        image_path, #path to the original image
    )

    imageSol = file_name.replace('.jpg', '.json')
    sol_path = folder_file_path('gdt', imageSol)
    with open(sol_path, 'r') as f:
        sol_data = json.load(f)
    sol_total = sol_data.get('total')
    # conver sol_total to float
    sol_total = re.sub(r'[^\d.]', '', str(sol_total))  # remove any non-numeric characters
    sol_total = float(sol_total) if sol_total else 0.0  # convert
    sol_date = sol_data.get('date')

    # append
    found_total_list.append(total)
    found_date_list.append(date)
    sol_total_list.append(sol_total)
    sol_date_list.append(sol_date)


# export to csv file
import pandas as pd
data = {
    'file_name': [f"{i:03}.jpg" for i in range(1, 100)],
    'found_total': found_total_list,
    'found_date': found_date_list,
    'sol_total': sol_total_list,
    'sol_date': sol_date_list
}
df = pd.DataFrame(data)
output_path = folder_file_path('data_out', 'results.csv')
df.to_csv(output_path, index=False)

