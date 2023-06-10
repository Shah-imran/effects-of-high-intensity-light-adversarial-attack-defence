import os
import uuid


def is_valid_line(line):
    data = line.strip().split()
    cls = int(data[0])
    return 0 <= cls <= 6

def remove_duplicates_and_invalids(lines):
    unique_lines = list(set(lines))  # remove duplicates
    return list(filter(is_valid_line, unique_lines))  # remove invalids

def rename_files_in_folder(root_dir):
    for dir_name in ['train', 'val']:
        img_dir = os.path.join(root_dir, dir_name, 'images')
        lbl_dir = os.path.join(root_dir, dir_name, 'labels')
        
        for img_file in os.listdir(img_dir):
            base_name, ext = os.path.splitext(img_file)
            lbl_file = f'{base_name}.txt'

            # Check if corresponding label file exists
            if lbl_file in os.listdir(lbl_dir):
                # Generate new unique name
                new_name = str(uuid.uuid4())
                new_img_file = f'{new_name}{ext}'
                new_lbl_file = f'{new_name}.txt'

                # Rename image file
                os.rename(os.path.join(img_dir, img_file), os.path.join(img_dir, new_img_file))
                # # Rename label file
                # os.rename(os.path.join(lbl_dir, lbl_file), os.path.join(lbl_dir, new_lbl_file))

                lbl_file_path = os.path.join(lbl_dir, lbl_file)
                new_lbl_file_path = os.path.join(lbl_dir, new_lbl_file)

                with open(lbl_file_path, 'r') as f:
                    lines = f.readlines()
                lines = remove_duplicates_and_invalids(lines)
                with open(new_lbl_file_path, 'w') as f:
                    f.writelines(lines)

                os.remove(lbl_file_path)  # remove old label file

# Usage
root_dir = './dataset'  # replace with your root directory
rename_files_in_folder(root_dir)
