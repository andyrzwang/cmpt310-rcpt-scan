import os

def give_colored_Image_full_File_Path(file_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))

    # project_root = project_root[0:-4:1]
    # print(project_root + '\n')

    roo_position = project_root.index('cmpt310-rcpt-scan')
    project_root = project_root[0:roo_position + len('cmpt310-rcpt-scan') + 1:1]
    # print(project_root + '\n')

    # path = project_root + '\data\dirty\large-receipt-image-dataset-SRD' + "\\" + file_name
    relative_path = os.path.join('data', 'dirty', 'large-receipt-image-dataset-SRD', file_name)
    path = os.path.normpath(os.path.join(project_root, relative_path))
    # print(path)
    return path


def folder_file_path(folderName, file_name):
    '''
    This function returns the full file path for a given file name in a specified folder.
    This only works for files inside data folder in this project

    'dirty' folder = not in use for clean images

    'images' folder = for text recognition

    'gdt' folder = for text label testing

    'info_data' folder = to test information about the images
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))

    # project_root = project_root[0:-4:1]
    # print(project_root + '\n')

    roo_position = project_root.index('cmpt310-rcpt-scan')
    project_root = project_root[0:roo_position + len('cmpt310-rcpt-scan') + 1:1]
    # print(project_root + '\n')

    # path = project_root + '\data\dirty\large-receipt-image-dataset-SRD' + "\\" + file_name

    if folderName == 'dirty':
        relative_path = os.path.join('data', 'dirty', 'large-receipt-image-dataset-SRD', file_name)
    
    else:
        relative_path = os.path.join('data', folderName, file_name)

    path = os.path.normpath(os.path.join(project_root, relative_path))
    # print(path)
    return path # string



# test code
# if __name__ == "__main__":
#     # test the function
#     file_name = '0001.jpg'
#     print(give_colored_Image_full_File_Path(file_name))

#     folderName = 'clean'
#     print(folder_file_path(folderName, file_name))
    
#     folderName = 'dirty'
#     print(folder_file_path(folderName, file_name))