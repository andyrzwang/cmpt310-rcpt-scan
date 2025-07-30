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

# for testing only
import sys
from file_full_path import folder_file_path
from PIL import Image
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def rescale(
    image_file,
    font_size_thresh: int = 14,
    small_scale: float = 2.0,
    large_scale: float = 1.0,
    interpolation=cv2.INTER_AREA
):
    """
    Dynamically resize width based on the smallest detected font size.
    
    Args:
      image_file: either a NumPy array (BGR or gray) or a tuple containing arrays.
      font_size_thresh: if smallest detected font < this, use small_scale; otherwise use large_scale.
      small_scale: multiplier to apply when fonts are small.
      large_scale: multiplier to apply when fonts are large.
      interpolation: OpenCV interpolation flag.
      
    Returns:
      Resized NumPy array.
    """
    # 1) Extract the first NumPy array if a tuple was passed
    if isinstance(image_file, tuple):
        image2 = next((x for x in image_file if isinstance(x, np.ndarray)), None)
    else:
        image2 = image_file

    # 2) Build a PIL image for pytesseract
    if len(image2.shape) == 2:  
        pil_img = Image.fromarray(image2)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    # 3) OCR to get word‐bounding‐box heights
    data = pyt.image_to_data(pil_img, output_type=pyt.Output.DICT)
    font_sizes = [
        data['height'][i]
        for i in range(len(data['text']))
        if data['text'][i].strip() != ''
    ]

    # 4) Determine multiplier
    smallest = min(font_sizes) if font_sizes else font_size_thresh
    multiplier = small_scale if smallest < font_size_thresh else large_scale

    # 5) Resize width only (keep original height)
    h, w = image2.shape[:2]
    new_w = int(w * multiplier)
    resized = cv2.resize(image2, (new_w, h), interpolation=interpolation)

    return resized


# # testing function
# if __name__ == "__main__":
#     for i in range(1, 20):
#             i = str(i)
#             file_number = i.zfill(3)
#             file_name = f"{file_number}.jpg"
#             image_path = folder_file_path('images', file_name)
#             if not os.path.exists(image_path):
#                 continue
#             image = cv2.imread(image_path)
#             scaled_image = rescale(image, i)
    # file_name = '1000-receipt.jpg'
    # full_file = give_colored_Image_full_File_Path(file_name)
    # print('++++++++++++++++++')
    # print(full_file)
    # # gray_image = grayscale(full_file)
    # image = cv2.imread(full_file)

    # scaled_image = rescale(image)

    # # testing
    # height, width = scaled_image.shape[:2]
    # print(height) #testing
    # print(width) # testing  

    # filesave = scaled_image
    # output_filename = '1000-receipt-scaled.jpg'
    # script_dir = os.path.dirname(os.path.abspath(__file__)) # current file's dir
    # output_path = os.path.join(script_dir, output_filename)

    # cv2.imwrite(output_path, filesave)

    # # output_filename = 'output_image.png'
    # # success = cv2.imwrite(output_filename, scaled_image)
    # # print(success)







