import os

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
    