import os
import shutil

# Directory paths
python_repo = r"C:\Users\Nouri\Documents\GitHub\Oneplusoneisone"
destination_folder = os.path.join(python_repo, "dashboards")

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

def find_and_copy_streamlit_dashboards(directory, destination):
    # Only look at the main directory (no recursion into subdirectories)
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path) and file.endswith(".py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "import streamlit" in content or "st." in content:
                        # Copy the file to the destination folder
                        shutil.copy(file_path, destination)
                        print(f"Copied: {file_path}")
            except UnicodeDecodeError:
                print(f"Skipped due to encoding error: {file_path}")

find_and_copy_streamlit_dashboards(python_repo, destination_folder)
print("Streamlit dashboards copied to:", destination_folder)
