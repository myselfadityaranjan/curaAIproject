import os
import pydicom
import numpy as np
from PIL import Image

def load_data(data_dir, img_size=(224, 224)):
    images = []
    labels = []

    for root, dirs, files in os.walk(data_dir):
        print(f"Checking directory: {root}")
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                print(f"Found file: {file_path}")
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

                # Optional: limit the number of images to manage memory
                if len(images) >= 500:  # Process 500 images at a time
                    print(f"Processing batch of {len(images)} images.")
                    yield np.array(images), labels
                    images, labels = [], []

    # Process any remaining images
    if images:
        print(f"Processing final batch of {len(images)} images.")
        yield np.array(images), labels

if __name__ == "__main__":
    data_dir = "/Users/adityaranjan/Documents/CURA AI/CURA AI training images/"
    # Load the data in batches
    for batch_images, batch_labels in load_data(data_dir):
        print(f"Batch loaded with {len(batch_images)} images and {len(batch_labels)} labels.")
