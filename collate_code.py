import os

# Function to convert all Python files in a directory into one text file
def convert_python_to_single_txt(directory, output_file):
    try:
        # Open the output file in write mode
        with open(output_file, "w", encoding="utf-8") as output_txt:
            # Loop through all files in the directory
            for filename in os.listdir(directory):
                # Check if the file is a Python file
                if filename.endswith(".py"):
                    # Construct full file path
                    python_file_path = os.path.join(directory, filename)

                    # Read the Python file content
                    with open(python_file_path, "r", encoding="utf-8") as py_file:
                        content = py_file.read()

                    # Write the content to the output text file
                    output_txt.write(f"# Start of {filename}\n")
                    output_txt.write(content)
                    output_txt.write(f"\n# End of {filename}\n\n")

                    print(f"Added: {filename} to {output_file}")

        print("All Python files have been merged into one text file.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Directory path containing Python files
directory_path = r"C:\\Users\\Nouri\\Documents\\GitHub\\Oneplusoneisone"
# Output file path
output_file_path = os.path.join(directory_path, "merged_python_files.txt")

convert_python_to_single_txt(directory_path, output_file_path)

