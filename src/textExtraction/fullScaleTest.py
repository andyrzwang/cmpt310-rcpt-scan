import cv2
import numpy as np
import os
import sys
import progressbar
import pytesseract as pyt
from scipy.ndimage import interpolation as inter
import pandas as pd

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





def full_scale_test(maxFile):
    file_name_list = []
    found_total_list = []
    found_date_list = []

    sol_total_list = []
    sol_date_list = []

    aCounter = 0

    with progressbar.ProgressBar(max_value=maxFile) as bar:
        

        for i in range(1, maxFile):
            bar.next()

            i = str(i)
            file_number = i.zfill(3)
            file_name = f"{file_number}.jpg"
            image_path = folder_file_path('images', file_name)
            if not os.path.exists(image_path):
                continue

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
            file_name_list.append(file_name)
            found_total_list.append(total)
            found_date_list.append(date)
            sol_total_list.append(sol_total)
            sol_date_list.append(sol_date)

            if total is None:
                total = 0.0
            if float(total) == float(sol_total):
                aCounter +=1

            

    # export to csv file
    
    data = {
        'file_name': file_name_list,
        'found_total': found_total_list,
        'found_date': found_date_list,
        'sol_total': sol_total_list,
        'sol_date': sol_date_list
    }
    df = pd.DataFrame(data)
    output_path = folder_file_path('dataOut', 'results.csv')
    df.to_csv(output_path, index=False)
    print("===============================================================")
    print(f"Total files processed: {len(file_name_list)}")
    print(f"Total files with correct total: {aCounter}")
    accuracy = (aCounter / len(file_name_list)) * 100 if file_name_list else 0
    print(f"Accuracy: {accuracy:.2f}%")
    print("================================================================")




if __name__ == "__main__":
    full_scale_test(200)