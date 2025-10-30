import os
from pathlib import Path

def remove_medium_from_filenames(folder_path):
    """
    Remove ' Medium' from the filenames of all images in the specified folder.

    Args:
        folder_path (str): Path to the folder containing the images.
    """
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    if not folder.is_dir():
        print(f"Error: The path '{folder_path}' is not a directory.")
        return

    # Process each file in the folder
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            if " Medium" in file.stem:
                # Generate the new filename
                new_name = file.stem.replace(" Medium", "") + file.suffix
                new_path = folder / new_name

                # Rename the file
                file.rename(new_path)
                print(f"Renamed: {file.name} -> {new_name}")

    print("Filename cleanup complete!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remove ' Medium' from image filenames in a folder.")
    parser.add_argument("--source", type=str, required=True, help="Path to the folder containing the images")

    args = parser.parse_args()

    remove_medium_from_filenames(args.source)