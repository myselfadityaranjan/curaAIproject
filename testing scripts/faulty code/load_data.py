import os
import pydicom

def load_data(data_dir):
    images = []
    labels = []

    for root, dirs, files in os.walk(data_dir):
        print(f"Checking directory: {root}")
        for file in files:
            if file.endswith(".dcm"):
                file_path = os.path.join(root, file)
                print(f"Found file: {file_path}")
                ds = pydicom.dcmread(file_path)
                images.append(ds.pixel_array)
                labels.append(ds.PatientID)  # Adjust this according to your label source

    print(f"Loaded {len(images)} images with labels.")
    return images, labels

if __name__ == "__main__":
    data_dir = "/Users/adityaranjan/Documents/CURA AI/CURA AI training images/"
    images, labels = load_data(data_dir)