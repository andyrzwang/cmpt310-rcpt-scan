import cv2
import os

def rescale(image_file):

    # get the height and weight of the image
    height, width = image_file.shape
    # print(height) #testing
    # print(width) # testing

    # calculate the aspect ratio
    aspect_ratio = width / height
    # print(aspect_ratio) #testing

    # set new width and new height
    new_width = 600 
    new_height = round(aspect_ratio * new_width)
    
    # resize the image with new height
    resized_image = cv2.resize(image_file, (new_width, new_height))

    #return
    return resized_image



# if __name__ == "__main__":
#     file_name = '1000-receipt.jpg'
#     full_file = give_colored_Image_full_File_Path(file_name)
#     print('++++++++++++++++++')
#     print(full_file)
#     gray_image = grayscale(full_file)
#     scaled_image = rescale(gray_image)

#     # testing
#     height, width = scaled_image.shape
#     print(height) #testing
#     print(width) # testing  

#     # output_filename = 'output_image.png'
#     # success = cv2.imwrite(output_filename, scaled_image)
#     # print(success)







