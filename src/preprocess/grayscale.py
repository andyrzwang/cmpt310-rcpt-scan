# Importing Needed libraries
import cv2

# Creating the grayscale funciton. 
# It will take in an input of an images file path, then 
# covert it to a grey scale image using the cv2 library 
# Lastly it will return the new grayscale image

# with CLAHE (Contrast Limited Adaptive Histogram Equalization)
'''def grayscale(image_file):

    # grayscale the image
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    
    # create clahe object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # apply clahe to the grayscale image
    enhanced = clahe.apply(gray)
    return enhanced
'''

def grayscale(image_file):

    # Converting image to grayscale
    # img_gray_mode = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_gray_mode = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)

    # Returning the new grayscale image
    return img_gray_mode