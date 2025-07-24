import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing the grayscale image conversion function from grayscale.py
# use the following for testing only
# from grayscale import grayscale
# from file_full_path import give_colored_Image_full_File_Path


def apply_binarization(image):
    # apply_binarization base on the grayscale image
    # using Otsu's thresholding method  


    # The first argument '0' indicates that the threshold value will be determined by Otsu's method.
    # The second argument '255' is the maximum value to use for the binary image.
    # cv2.THRESH_BINARY ensures a binary output (0 or 255).
    # cv2.THRESH_OTSU is the flag that activates Otsu's method.
    ret, otsu_thresholded_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ret is the threshold value calculated by the Otsu's method.
    # for debugging purposes
    # print(f"Otsu's thresholding value: {ret}")


    return otsu_thresholded_img







# # test function to verify if an image has been successfully converted into grayscale
# # must comment out the test function (main)

# if __name__ == "__main__":
#     # Example usage
#     image_path = give_colored_Image_full_File_Path('1005-receipt.jpg')
#     grayscale_image = grayscale(image_path)


#     # call function
#     binarized_image = apply_binarization(grayscale_image)

#     # store image

#     filesave = binarized_image
#     output_filename = '1000-receipt-binary.jpg'
#     script_dir = os.path.dirname(os.path.abspath(__file__)) # current file's dir
#     output_path = os.path.join(script_dir, output_filename)

#     cv2.imwrite(output_path, filesave)

