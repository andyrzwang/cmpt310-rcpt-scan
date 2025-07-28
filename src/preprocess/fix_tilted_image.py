import cv2
import numpy as np
import os
import sys
import pytesseract as pyt
from scipy.ndimage import interpolation as inter

from file_full_path import folder_file_path

def fix_tilted_image(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(image, angle)
        scores.append(score)
    
    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected



# # test functions
# if __name__ == "__main__":
#     file_name = '022.jpg'
#     full_file = folder_file_path('images', file_name)
#     print('++++++++++++++++++')
#     # print(full_file)

#     image = cv2.imread(full_file)
#     if image is None:
#         print("Error: Image not found.")
#     else:
#         best_angle, corrected_image = fix_tilted_image(image)

#         # Save the corrected image
#         output_filename = '1000-receipt-corrected.jpg'
#         script_dir = os.path.dirname(os.path.abspath(__file__))  # current file's dir
#         output_path = os.path.join(script_dir, output_filename)

#         cv2.imwrite(output_path, corrected_image)
#         print(f"Corrected image saved as {output_path}")