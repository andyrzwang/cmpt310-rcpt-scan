
import cv2

def greyscale(file_path:str):
    
    img_gray_mode = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    return img_gray_mode

