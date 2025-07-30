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

def grayscale(image_file, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    - clip_limit: threshold for contrast limiting (float)
    - tile_grid_size: size of grid for CLAHE (tuple of two ints)
    """
    gray = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)
