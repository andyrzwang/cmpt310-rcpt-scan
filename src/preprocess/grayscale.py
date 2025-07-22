# Importing Needed libraries
import cv2

# Creating the grayscale funciton. 
# It will take in an input of an images file path, then 
# covert it to a grey scale image using the cv2 library 
# Lastly it will return the new grayscale image
def grayscale(file_path:str):

    # Converting image to grayscale
    img_gray_mode = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Returning the new grayscale image
    return img_gray_mode

