import cv2
import numpy as np
import os
import sys
import pytesseract as pyt
from scipy.ndimage import interpolation as inter
from PIL import Image
import re
from datetime import datetime
import json
import pandas as pd   

# import preprocess modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocess')))
from file_full_path import folder_file_path
from grayscale import grayscale
from binarization import apply_binarization
from rescale import rescale
from fix_tilted_image import fix_tilted_image

def image_preProcess(image):
    # tuned hyper-parameters from CV:
    gray = grayscale(image, clip_limit=2.0, tile_grid_size=(16,16))
    bin_img = apply_binarization(gray, method='otsu', block_size=21, C=5)

    # inject rescale step into the production pipeline
    res = rescale(
        bin_img,
        font_size_thresh=16,
        small_scale=1.5,
        large_scale=1.0
    )
    #upRight = fix_tilted_image(bin)
    return res

def findTotal(lines, ocr_data):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(current_dir, 'fakeTotal.json')
    true_path = os.path.join(current_dir, 'trueTotal.json')

    with open(fake_path, 'r') as f:
        fake_data = json.load(f)
    with open(true_path, 'r') as f:
        true_data = json.load(f)

    fake_total_list = fake_data.get('fakeTotalList', [])
    true_total_list = true_data.get('trueTotalList', [])

    GuessedTotal = []
    for i in range(len(lines)):
        line = lines[i]
        upper_line = line.upper()
        if any(true_kw in upper_line for true_kw in true_total_list):
            if not any(fake_kw in upper_line for fake_kw in fake_total_list):
                matches = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d{2})|\$\d+(?:\.\d{2})?', line)
                if matches:
                    cleaned = re.sub(r'[^\d\.]', '', matches[0])
                    total_value = float(cleaned)
                    GuessedTotal.append(total_value)

    return max(GuessedTotal) if GuessedTotal else None

def findDate(lines, ocr_data):
    return None

def text_detection(image, original_image, image_file_Path):
    if isinstance(image, tuple):
        image = next((x for x in image if isinstance(x, np.ndarray)), None)

    text_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if len(image.shape) == 2:
        text_image = Image.fromarray(image)
    elif len(image.shape) == 3:
        text_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    ocr_data = pyt.image_to_data(text_image, output_type=pyt.Output.DICT)
    ocr_data = pd.DataFrame(ocr_data)

    # ensure every entry is a string (fill NaNs) so .str works
    ocr_data['text'] = ocr_data['text'].fillna('').astype(str)
    ocr_data = ocr_data[ocr_data['text'].str.strip() != '']

    lines = (
        ocr_data.groupby(['block_num', 'line_num'])['text']
        .apply(lambda words: ' '.join(words.tolist()))
        .tolist()
    )

    total = findTotal(lines, ocr_data)
    date = findDate(lines, ocr_data)

    return total, date

if __name__ == "__main__":
    file_name = '015.jpg'
    image_path = folder_file_path('images', file_name)
    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist.")
        sys.exit(1)

    image = cv2.imread(image_path)
    pre_processed_image = image_preProcess(image)

    total, date = text_detection(
        pre_processed_image,
        pre_processed_image,
        image_path,
    )

    imageSol = file_name.replace('.jpg', '.json')
    sol_path = folder_file_path('gdt', imageSol)
    with open(sol_path, 'r') as f:
        sol_data = json.load(f)

    sol_total = sol_data.get('total')
    sol_total = re.sub(r'[^\d.]', '', str(sol_total))
    sol_total = float(sol_total) if sol_total else 0.0
    sol_date = sol_data.get('date')

    print(f"Total: {total}, Date: {date}")
    print(f"Solution Total: {sol_total}, Solution Date: {sol_date}")

    if float(total) == float(sol_total):
        print("The total value is correct.")
    else:
        print("The total value is incorrect.")
