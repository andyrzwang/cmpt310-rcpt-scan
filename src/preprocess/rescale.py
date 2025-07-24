import cv2
import os


# # for testing only
# import sys
# from file_full_path import give_colored_Image_full_File_Path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def rescale(image_file):

    # get the height and width of the image
    
    height, width = image_file.shape[:2]
    # print(height) #testing
    # print(width) # testing

    # calculate the aspect ratio
    aspect_ratio = height / width
    # print(aspect_ratio) #testing

    # set new width and new height
    new_width = 600 
    new_height = round(aspect_ratio * new_width)
    
    # resize the image with new height
    resized_image = cv2.resize(image_file, (new_width, new_height))

    #return
    return resized_image


# testing function
# if __name__ == "__main__":
#     file_name = '1000-receipt.jpg'
#     full_file = give_colored_Image_full_File_Path(file_name)
#     print('++++++++++++++++++')
#     print(full_file)
#     # gray_image = grayscale(full_file)
#     image = cv2.imread(full_file)

#     scaled_image = rescale(image)

#     # testing
#     height, width = scaled_image.shape[:2]
#     print(height) #testing
#     print(width) # testing  

#     filesave = scaled_image
#     output_filename = '1000-receipt-scaled.jpg'
#     script_dir = os.path.dirname(os.path.abspath(__file__)) # current file's dir
#     output_path = os.path.join(script_dir, output_filename)

#     cv2.imwrite(output_path, filesave)

#     # output_filename = 'output_image.png'
#     # success = cv2.imwrite(output_filename, scaled_image)
#     # print(success)







