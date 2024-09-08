import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
from sklearn.preprocessing import LabelEncoder

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Get the image as a NumPy array
        image_array = self.images[idx]
        label = self.labels[idx]
        
        # Convert NumPy array to PIL Image
        image = transforms.ToPILImage()(image_array)  # Convert to PIL Image
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_data_from_files(images_path='images.npy', labels_path='labels.npy'):
    print("Loading data from files...")
    images = np.load(images_path, mmap_mode='r')  # Use memory-mapped mode
    labels = np.load(labels_path)
    print(f"Loaded {len(labels)} labels and corresponding images.")
    return images, labels

def encode_labels(labels):
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    print("Labels encoded successfully.")
    return encoded_labels, label_encoder

def train_model(train_loader, val_loader, num_epochs=10, num_classes=2):
    print("Initializing the model...")
    model = models.resnet18(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print("Model initialized.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            print(f"Batch {i}/{len(train_loader)} processed with loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs} completed. Training Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader, 1):
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(f"Validation Batch {i}/{len(val_loader)} processed with loss: {loss.item():.4f}")

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

    torch.save(model.state_dict(), 'model.pth')
    print('Model training completed and saved to model.pth')

if __name__ == "__main__":
    # Load data from .npy files
    images, labels = load_data_from_files('images.npy', 'labels.npy')
    
    # Encode labels
    labels_encoded, label_encoder = encode_labels(labels)
    
    # Determine the number of unique classes
    num_classes = len(np.unique(labels_encoded))
    
    # Split data
    train_size = 55312
    val_size = 11853

    X_train = images[:train_size]
    y_train = labels_encoded[:train_size]
    X_val = images[train_size:train_size + val_size]
    y_val = labels_encoded[train_size:train_size + val_size]

    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # Resize image to 224x224
        transforms.ToTensor(),             # Convert PIL Image to tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Create datasets
    print("Creating datasets...")
    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    val_dataset = CustomDataset(X_val, y_val, transform=transform)

    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Increased batch size
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)    # Increased batch size

    # Start model training
    print("Starting model training...")
    train_model(train_loader, val_loader, num_epochs=10, num_classes=num_classes)
