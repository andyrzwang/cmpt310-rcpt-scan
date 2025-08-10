# Importing Needed libraries
import cv2

# Creating the grayscale funciton. 
# It will take in an input of an images file path, then 
# covert it to a grey scale image using the cv2 library 
# Lastly it will return the new grayscale image
'''
def grayscale(image_file):

    # Converting image to grayscale
    # img_gray_mode = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img_gray_mode = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)

    # Returning the new grayscale image
    return img_gray_mode
'''
def grayscale(image_file, clip_limit=2.0, tile_grid_size=(16, 16)):
    """
    - clip_limit: threshold for contrast limiting (float)
    - tile_grid_size: size of grid for CLAHE (tuple of two ints)
    """
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)