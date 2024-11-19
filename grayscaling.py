# from PIL import Image
# import os
# import csv
# import numpy as np
# import pillow_heif  # Enables HEIC/HEIF support

# # Define the dataset directory
# dataset_dir = r"C:\Users\IdeaPad\Downloads\bisindo-dataset-noise"

# # CSV file to save the pixel data
# csv_file = "dataset_noise.csv"

# # Open the CSV file for writing
# with open(csv_file, mode="w", newline="") as file:
#     writer = csv.writer(file)
    
#     # Write the header (label, filename, pixel_1 to pixel_784)
#     writer.writerow(["label", "filename"] + [f"pixel_{i}" for i in range(1, 785)])
    
#     # Loop through the dataset folder
#     for subdir, dirs, files in os.walk(dataset_dir):
#         for folder_name in sorted(dirs):  # Process subdirectories alphabetically
#             print(f"Processing {folder_name}")
#             folder_path = os.path.join(subdir, folder_name)
#             if folder_name == 'NOTHING':
#                 label = 26
#             else:
#                 label = ord(folder_name.upper()) - ord('A')  # Convert folder name to label (A=0, B=1, ..., Z=25)
#             if 0 <= label <= 26:  # Ensure the folder is A-Z
#                 for filename in os.listdir(folder_path):
#                     if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif')):
#                         try:
#                             # Load the image
#                             img_path = os.path.join(folder_path, filename)
#                             if filename.lower().endswith(('.heic', '.heif')):
#                                 heif_file = pillow_heif.read_heif(img_path)
#                                 image = Image.frombytes(
#                                     heif_file.mode, heif_file.size, heif_file.data
#                                 )
#                             else:
#                                 image = Image.open(img_path)
                            
#                             # Process the image
#                             image = image.convert('L').resize((28, 28))  # Grayscale and resize to 28x28
#                             pixels = np.array(image).flatten()  # Flatten the image into a 1D array
                            
#                             # Write label and pixel data to CSV
#                             writer.writerow([label, filename] + pixels.tolist())
#                         except Exception as e:
#                             print(f"Error processing {img_path}: {e}")

# print(f"Image pixel data saved to {csv_file}")

import cv2
import numpy as np
import os
import csv
from PIL import Image
import pillow_heif  # Enables HEIC/HEIF support

# Define the dataset directory
dataset_dir = r"C:\Users\IdeaPad\Downloads\dataset2\Dataset BISINDO\datatrain"

# Define the output directory for CLAHE images
output_dir = r"C:\Users\IdeaPad\Documents\ML\BISINDO_GitHub\MLFinalProject\dataset2_clahe"
os.makedirs(output_dir, exist_ok=True)

# CSV file to save the pixel data
csv_file = "dataset_noise_dataset2_CLAHE.csv"

# Open the CSV file for writing
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    
    # Write the header (label, filename, pixel_1 to pixel_784)
    writer.writerow(["label", "filename"] + [f"pixel_{i}" for i in range(1, 785)])
    
    # Loop through the dataset folder
    for subdir, dirs, files in os.walk(dataset_dir):
        for folder_name in sorted(dirs):  # Process subdirectories alphabetically
            print(f"Processing {folder_name}")
            folder_path = os.path.join(subdir, folder_name)
            output_folder_path = os.path.join(output_dir, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            
            if folder_name == 'NOTHING':
                label = 26
            else:
                label = ord(folder_name.upper()) - ord('A')  # Convert folder name to label (A=0, B=1, ..., Z=25)
            if 0 <= label <= 26:  # Ensure the folder is A-Z
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.heif')):
                        try:
                            # Load the image
                            img_path = os.path.join(folder_path, filename)
                            if filename.lower().endswith(('.heic', '.heif')):
                                heif_file = pillow_heif.read_heif(img_path)
                                pil_image = Image.frombytes(
                                    heif_file.mode, heif_file.size, heif_file.data
                                )
                            else:
                                pil_image = Image.open(img_path)
                            
                            # Convert PIL image to OpenCV format
                            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                            
                            # Apply CLAHE
                            image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            clahe = cv2.createCLAHE(clipLimit=5)
                            processed_image = clahe.apply(image_bw)
                            
                            # Save CLAHE image
                            clahe_image_path = os.path.join(output_folder_path, filename)
                            cv2.imwrite(clahe_image_path, processed_image)
                            
                            # Resize to 28x28 for compatibility
                            resized_image = cv2.resize(processed_image, (28, 28))
                            
                            # Flatten the image into a 1D array
                            pixels = resized_image.flatten()
                            
                            # Write label and pixel data to CSV
                            writer.writerow([label, filename] + pixels.tolist())
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")

print(f"CLAHE-processed images saved to {output_dir}")
print(f"Image pixel data with CLAHE saved to {csv_file}")
