import os
import pydicom
import numpy as np
from PIL import Image

def load_data(data_dir, img_size=(224, 224)):
    images = []
    labels = []
    file_count = 0

    for root, dirs, files in os.walk(data_dir):
        print(f"Checking directory: {root}")
        print(f"Found {len(files)} files in this directory.")
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                print(f"Found file: {file_path}")
                file_count += 1
                try:
                    ds = pydicom.dcmread(file_path)
                    
                    # Convert the DICOM image to a NumPy array
                    img_array = ds.pixel_array

                    # Resize the image to 224x224
                    img = Image.fromarray(img_array)
                    img = img.convert("RGB")  # Convert to 3-channel
                    img = img.resize(img_size, Image.Resampling.LANCZOS)  # Use Resampling.LANCZOS
                    img_array_resized = np.array(img)

                    # Append image to list
                    images.append(img_array_resized)

                    # Placeholder for labels, assuming PatientID is used as a placeholder
                    labels.append(ds.PatientID)  # Adjust this according to your label source
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Total files processed: {file_count}")
    # Return all images and labels without batching
    return np.array(images), labels

if __name__ == "__main__":
    data_dir = "/Users/adityaranjan/Documents/CURA AI/CURA AI training images/"
    images, labels = load_data(data_dir)
    print(f"Loaded {len(images)} images with {len(labels)} labels.")
