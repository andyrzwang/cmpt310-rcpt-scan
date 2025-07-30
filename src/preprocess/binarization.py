import cv2
import os
import sys

# # Importing the grayscale image conversion function from grayscale.py
# # use the following for testing only
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from grayscale import grayscale
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from file_full_path import give_colored_Image_full_File_Path 


def apply_binarization(image, method='otsu', block_size=21, C=5):
    """
    image: single-channel (grayscale) numpy array
    method: 'otsu' or 'adaptive'
    block_size, C: only used for adaptive thresholding
    """
    if method == 'otsu':
        blurred = cv2.GaussianBlur(image, (5,5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == 'adaptive':
        # block_size must be odd
        bs = block_size if block_size % 2 == 1 else block_size + 1
        thresh = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            C
        )
    else:
        raise ValueError(f"Unknown binarization method: {method}")
    return thresh







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

