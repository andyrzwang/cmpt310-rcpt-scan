import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import sys


from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from file_full_path import give_colored_Image_full_File_Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from grayscale import grayscale
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rescale import rescale

from binarization import apply_binarization



def rescale2(image):
    width, height = image.shape[:2]

    new_width = 600
    new_height = int((new_width / width) * height)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def main(image, original):

    # Receipt contour detection
    
    # gray scale
    gray_image = grayscale(image)

    # rescale +++++++++++++++++++++++++++++++
    scaled_image = rescale(gray_image)

    original_scale = rescale(original)
    original = original_scale
    
    # binarization
    ret, binarized_image = cv2.threshold(scaled_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5,5), np.uint8)
    erosion = cv2.erode(binarized_image, kernel, iterations = 1)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)


    # remove noise
    blurred = cv2.GaussianBlur(opening, (5, 5), 0)

    # detect white pixels and regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # high light white regions
    dilated = cv2.dilate(blurred, rectKernel)
    # identify and draw edges
    edged = cv2.Canny(dilated, 100, 200, apertureSize = 7)


    # Detect all contours in Canny-edged image
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # highlight the contours and store them in a new image
    # image_with_contours = cv2.drawContours(original.copy(), contours, -1, (0,0,255), 3)
    image_with_contours = cv2.drawContours(opening.copy(), contours, -1, (0,0,255), 3)     ## which should I use? original or opening?


    ## +++++++++++++++++++++++++
    # find the contour for the receipt
    # Add image border to help detect receipts touching the edge
    border_size = 10
    bordered = cv2.copyMakeBorder(image_with_contours, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=255)
    bordered_blurred = cv2.GaussianBlur(bordered, (5, 5), 0)
    bordered_dilated = cv2.dilate(bordered_blurred, rectKernel)
    bordered_edged = cv2.Canny(bordered_dilated, 120, 200, apertureSize=3)
    bordered_contours, _ = cv2.findContours(bordered_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Shift contours back to original coordinates
    contours = []
    for c in bordered_contours:
        c = c - [border_size, border_size]
        contours.append(c)
    
    ## +++++++++++++++++++++++++


    receipt_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.035 * peri, True)

        # if our approximated contour has four points, we can assume it is receipt's rectangle
        if len(approx) == 4:
            receipt_contour = approx
    
    image_with_receipt_contour = cv2.drawContours(original.copy(), [receipt_contour], -1, (0, 0, 255), 2)

    print("333")
    # save file - - - - CHANGE HERE
    filesave = image_with_receipt_contour
    output_filename = '1000-receipt-binary.jpg'
    script_dir = os.path.dirname(os.path.abspath(__file__)) # current file's dir
    output_path = os.path.join(script_dir, output_filename)

    cv2.imwrite(output_path, filesave)


    print("444")
    




if __name__ == "__main__":
    imageName = "1005-receipt.jpg"
    file_name = give_colored_Image_full_File_Path(imageName)
    image = cv2.imread(file_name)
    print("1")
    main(image, image)

    print("Segmentation completed. Check the output file in the same directory as this script.")