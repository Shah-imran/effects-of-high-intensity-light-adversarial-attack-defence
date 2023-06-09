
import os
import glob

# replace /path/to/directory with your actual directory
dir_path = os.path.join(os.getcwd(), "runs","experiment","random")
print(dir_path)

# Iterate over each sub-directory
for subdir in os.listdir(dir_path):
    subdir_path = os.path.join(dir_path, subdir)
    
    # Ensure it's a directory
    if os.path.isdir(subdir_path):
        # Iterate over each PNG file in the sub-directory
        for file_path in glob.glob(os.path.join(subdir_path, "random*.png")):
            # Replace 'mod' with 'image' in the filename
            directory, filename = os.path.split(file_path)
            # Replace 'mod' with 'image' in the filename
            new_filename = filename.replace('random', 'image')
            # Combine the directory part and the new filename
            new_file_path = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(file_path, new_file_path)
