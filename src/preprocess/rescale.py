import cv2
import os

# pre test
def give_colored_Image_full_File_Path(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))

    # project_root = project_root[0:-4:1]
    print(project_root + '\n')

    roo_position = project_root.index('cmpt310-rcpt-scan')
    project_root = project_root[0:roo_position + len('cmpt310-rcpt-scan') + 1:1]
    print(project_root + '\n')

    # path = project_root + '\data\dirty\large-receipt-image-dataset-SRD' + "\\" + file_name
    relative_path = os.path.join('data', 'dirty', 'large-receipt-image-dataset-SRD', file_name)
    path = os.path.normpath(os.path.join(project_root, relative_path))
    print(path)
    return path

def grayscale(file_path:str):

    # Converting image to grayscale
    img_gray_mode = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Returning the new grayscale image
    return img_gray_mode


def rescale(image_file):

    # get the height and weight of the image
    height, width = image_file.shape
    print(height) #testing
    print(width) # testing

    # calculate the aspect ratio
    aspect_ratio = width / height
    print(aspect_ratio) #testing

    # set new width and new height
    new_width = 600 
    new_height = round(aspect_ratio * new_width)
    
    # resize the image with new height
    resized_image = cv2.resize(image_file, (new_width, new_height))

    #return
    return resized_image



if __name__ == "__main__":
    file_name = '1000-receipt.jpg'
    full_file = give_colored_Image_full_File_Path(file_name)
    print('++++++++++++++++++')
    print(full_file)
    gray_image = grayscale(full_file)
    scaled_image = rescale(gray_image)

    # testing
    height, width = scaled_image.shape
    print(height) #testing
    print(width) # testing  

    # output_filename = 'output_image.png'
    # success = cv2.imwrite(output_filename, scaled_image)
    # print(success)







