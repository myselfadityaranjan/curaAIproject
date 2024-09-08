import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from load_data import load_data

def save_data(images, labels, images_path='images.npy', labels_path='labels.npy'):
    print("Saving data...")
    np.save(images_path, images)
    np.save(labels_path, labels)
    print(f"Data saved to {images_path} and {labels_path}")

def split_data(images, labels):
    print("Splitting data into training, validation, and test sets...")
    train_size = 55312
    val_size = 11853
    test_size = 11853

    X_train = images[:train_size]
    y_train = labels[:train_size]
    X_val = images[train_size:train_size + val_size]
    y_val = labels[train_size:train_size + val_size]
    X_test = images[train_size + val_size:]
    y_test = labels[train_size + val_size:]

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=16):
    print("Creating data loaders...")

    # Fit label encoder on all labels
    label_encoder = LabelEncoder()
    all_labels = np.concatenate((y_train, y_val, y_test))
    label_encoder.fit(all_labels)

    # Encode labels
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.int64)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.int64)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.int64)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Training loader created with batch size {batch_size}")
        print(f"Validation loader created with batch size {batch_size}")
        print(f"Test loader created with batch size {batch_size}")

        return train_loader, val_loader, test_loader
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return None, None, None

if __name__ == "__main__":
    data_dir = "/Users/adityaranjan/Documents/CURA AI/CURA AI training images/"
    
    print("Loading data...")
    images, labels = load_data(data_dir)
    
    save_data(images, labels)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)
    
    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("All steps completed successfully.")
