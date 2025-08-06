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


def rescale(image_file):
    # Convert image_file from cv2 (numpy array) format to PIL Image for pytesseract

    if isinstance(image_file, tuple):
        # Pick the first valid NumPy array
        image2 = next((x for x in image_file if isinstance(x, np.ndarray)), None)
    else:
        image2 = image_file

    text_image = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    if len(image2.shape) == 2:  # grayscale
        text_image = Image.fromarray(image2)
    elif len(image2.shape) == 3:  # BGR color
        text_image = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    data = pyt.image_to_data(text_image, output_type=pyt.Output.DICT)

    font_sizes_in_pixels = []
    for i in range(len(data['text'])):
        if data['text'][i].strip() != '':  # Ensure it's not an empty string
            height = data['height'][i]
            font_sizes_in_pixels.append(height)
    
    smallest_font_size = min(font_sizes_in_pixels)

    # get the height and width of the image
    
    height, width = image2.shape[:2]
    # print(f"Name: {id}, Height: {height}, Width: {width}")

    if smallest_font_size < 14:
        multiplier = 2
    else:
        multiplier = 1

    # set new width and new height
    new_width = width * multiplier 
    new_height = height
    
    # resize the image with new height
    resized_image = cv2.resize(image2, (new_width, new_height))

    #return
    return resized_image


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







