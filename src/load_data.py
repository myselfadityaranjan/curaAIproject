import os
import pydicom
import numpy as np
from PIL import Image
from sklearn.utils import shuffle

def load_data(data_dir, img_size=(224, 224), max_images=25600): #function to preprocess .dcm files from dataset directory
    images = []
    labels = []
    file_count = 0 #track how many files processed during process

    for root, dirs, files in os.walk(data_dir): #use oswalk to go through directory tree, stopping at .dcm files
        print(f"Checking directory: {root}")
        print(f"Found {len(files)} files in this directory.")
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                print(f"Found file: {file_path}")
                file_count += 1
                try:
                    ds = pydicom.dcmread(file_path)
                    
                    # convert the .dcm image to .npy array
                    img_array = ds.pixel_array

                    # resize the image to 224x224
                    img = Image.fromarray(img_array)
                    img = img.convert("RGB")  # convert to 3-channel for cnn
                    img = img.resize(img_size, Image.Resampling.LANCZOS) 
                    img_array_resized = np.array(img)

                    # add processed image to list
                    images.append(img_array_resized)

                    # using patientid as placeholder label, adjust this if needed
                    labels.append(ds.PatientID) 
                    
                    if len(images) >= max_images:
                        break
                except Exception as e:
                    print(f"Error reading {file_path}: {e}") #for errors with file reading
            if len(images) >= max_images:
                break
        if len(images) >= max_images:
            break

    print(f"Total files processed: {file_count}") #summary
    print(f"Total images loaded: {len(images)}")
    
    # finally convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # shuffle dataset
    images, labels = shuffle(images, labels, random_state=42)
    
    return images, labels

if __name__ == "__main__":
    data_dir = "/Users/adityaranjan/Documents/CURA AI/CURA AI training images/"
    images, labels = load_data(data_dir)
    print(f"Loaded {len(images)} images with {len(labels)} labels.")
