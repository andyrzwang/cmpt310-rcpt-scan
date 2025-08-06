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

# ─── Import Preprocessing Modules ─────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocess')))
from file_full_path import folder_file_path
from grayscale import grayscale
from binarization import apply_binarization
from rescale import rescale
from fix_tilted_image import fix_tilted_image

# ─── Full Preprocessing Pipeline ──────────────────────────────────────────────
def image_preProcess(image):
    gray = grayscale(image)
    bin_img = apply_binarization(gray)
    upRight = fix_tilted_image(bin_img)
    return upRight

# ─── Draw Bounding Boxes ──────────────────────────────────────────────────────
def draw_text_boxes(image, ocr_data, output_path="output_with_boxes.png"):
    if ocr_data is None or ocr_data.empty:
        print("No OCR data to draw.")
        return

    # Ensure image is in color (BGR) for red boxes
    if len(image.shape) == 2:
        annotated_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated_img = image.copy()

    for _, row in ocr_data.iterrows():
        x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
        text = str(row['text'])
        if text.strip():
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box
            cv2.putText(annotated_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, annotated_img)
    print(f"Saved bounding box image to {output_path}")

# ─── Total Extraction ─────────────────────────────────────────────────────────
def findTotal(lines, ocr_data):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fake_path = os.path.join(script_dir, 'fakeTotal.json')
    true_path = os.path.join(script_dir, 'trueTotal.json')

    with open(fake_path, 'r') as f:
        fake_data = json.load(f)
    with open(true_path, 'r') as f:
        true_data = json.load(f)

    fake_total_list = fake_data.get('fakeTotalList', [])
    true_total_list = true_data.get('trueTotalList', [])

    GuessedTotal = []
    for line in lines:
        upper_line = line.upper()
        if any(true_kw in upper_line for true_kw in true_total_list):
            if not any(fake_kw in upper_line for fake_kw in fake_total_list):
                matches = re.findall(r'\d{1,3}(?:,\d{3})*(?:\.\d{2})|\$\d+(?:\.\d{2})?', line)
                if matches:
                    cleaned = re.sub(r'[^\d\.]', '', matches[0])
                    try:
                        total_value = float(cleaned)
                        GuessedTotal.append(total_value)
                    except:
                        continue

    return max(GuessedTotal) if GuessedTotal else None

# ─── Date Extraction (stub) ───────────────────────────────────────────────────
def findDate(lines, ocr_data):
    return None

# ─── OCR + Total Detection ────────────────────────────────────────────────────
def text_detection(image, image_path_for_boxes="output_with_boxes.png"):
    if isinstance(image, tuple):
        image = next((x for x in image if isinstance(x, np.ndarray)), None)

    text_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if len(image.shape) == 2:
        text_image = Image.fromarray(image)

    ocr_data = pyt.image_to_data(text_image, output_type=pyt.Output.DATAFRAME)
    ocr_data = ocr_data.dropna(subset=['text'])
    ocr_data = ocr_data[ocr_data['text'].str.strip() != '']

    lines = (
        ocr_data.groupby(['block_num', 'line_num'])['text']
        .apply(lambda words: ' '.join(words.tolist()))
        .tolist()
    )

    total = findTotal(lines, ocr_data)
    date = findDate(lines, ocr_data)

    # Save bounding boxes
    draw_text_boxes(image, ocr_data, output_path=image_path_for_boxes)

    return total, date

# ─── Manual Test Execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    file_name = '010.jpg'
    image_path = folder_file_path('images', file_name)
    if not os.path.exists(image_path):
        print(f"File {image_path} does not exist.")
        sys.exit(1)

    image = cv2.imread(image_path)
    pre_processed_image = image_preProcess(image)

    total, date = text_detection(pre_processed_image, image_path_for_boxes="output_with_boxes.png")

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

    if total is None:
        print("The total value could not be detected.")
    elif float(total) == float(sol_total):
        print("The total value is correct.")
    else:
        print("The total value is incorrect.")
