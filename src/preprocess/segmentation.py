import numpy as np
import cv2
import matplotlib.pyplot as plt


from PIL import Image
def rescale(image_file):

    # get the height and weight of the image
    height, width = image_file.shape[:2]
    print(height) #testing
    print(width) # testing

    # calculate the aspect ratio
    aspect_ratio = width / height
    print(aspect_ratio) #testing

    # set new width and new height
    new_width = 600 
    new_height = round(aspect_ratio * new_width)
    
    # resize the image with new height
    resized_image = cv2.resize(image_file, (new_width, new_height))

    #return
    return resized_image


def grayscale(file_path:str):

    # Converting image to grayscale
    img_gray_mode = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Returning the new grayscale image
    return img_gray_mode


def main():
    file_name = 'C:\workspace\SFUschool\cmpt310\cmpt310-rcpt-scan\data\dirty\large-receipt-image-dataset-SRD\\1000-receipt.jpg'

    # gray_image = grayscale(file_name)
    # scaled_image = rescale(gray_image)


    
    
    return null





