import os
import shutil


def unpack_subfolders(main_folder):
    for item in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, item)
        if os.path.isdir(subfolder_path):  # Check if it's a sub-folder
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                if os.path.isfile(file_path):  # Check if it's a file
                    destination_path = os.path.join(main_folder, file)
                    # Handle overwrites by just deleting the copy
                    if not os.path.exists(destination_path):
                        shutil.move(file_path, destination_path)  # Move file to main folder
                    else:
                        print(f"File {file} already exists in the main folder.")
                        os.remove(file_path)
            # Remove the sub-folder after moving its contents
            # if not os.listdir(subfolder_path):
            #     print(f"Removing sub-folder: {subfolder_path}")
            #     shutil.rmtree(subfolder_path)

            os.rmdir(subfolder_path)  # Remove the empty sub-folder


# Example usage
main_folder = input("Enter the path to the main folder: ")
unpack_subfolders(main_folder)
