import cv2
import os
import sys

# # Importing the grayscale image conversion function from grayscale.py
# # use the following for testing only
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from grayscale import grayscale
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from file_full_path import give_colored_Image_full_File_Path 


def apply_binarization(image):
    # apply_binarization base on the grayscale image
    # using Otsu's thresholding method  


    # Step 1: Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    contrast_img = clahe.apply(image)

    # Step 2: Optional Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(contrast_img, (3, 3), 0)

    # Step 3: Apply Otsu's thresholding
    ret, otsu_thresholded_img = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Optional debug
    # print(f"Otsu's threshold value: {ret}")

    return otsu_thresholded_img







# # test function to verify if an image has been successfully converted into grayscale
# # must comment out the test function (main)

# if __name__ == "__main__":
#     # Example usage
#     image_path = give_colored_Image_full_File_Path('1000-receipt.jpg')

#     print(f"Input image path: {image_path}")  # Debug print

#     # Read the image
#     image_file = cv2.imread(image_path)


#     grayscale_image = grayscale(image_file)


#     # call function
#     binarized_image = apply_binarization(grayscale_image)

#     # store image

#     filesave = binarized_image
#     output_filename = '1000-receipt-binary.jpg'
#     script_dir = os.path.dirname(os.path.abspath(__file__)) # current file's dir
#     output_path = os.path.join(script_dir, output_filename)

#     cv2.imwrite(output_path, filesave)

