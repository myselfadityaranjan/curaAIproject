import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
from sklearn.preprocessing import LabelEncoder

class CustomDataset(Dataset): #customm dataset class to handle images/labels
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels) #return sample number

    def __getitem__(self, idx):
        # get the image as a npy array
        image_array = self.images[idx]
        label = self.labels[idx]
        
        # convert npy array to PIL Image
        image = transforms.ToPILImage()(image_array) 
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_data_from_files(images_path='images.npy', labels_path='labels.npy'):
    print("Loading data from files...")
    images = np.load(images_path, mmap_mode='r')  # use memory-mapped mode > faster (less memory heavy on system)
    labels = np.load(labels_path)
    print(f"Loaded {len(labels)} labels and corresponding images.")
    return images, labels

def encode_labels(labels): #encode string labels to numerical values
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    print("Labels encoded successfully.")
    return encoded_labels, label_encoder

def train_model(train_loader, val_loader, num_epochs=10, num_classes=2): #to train model
    print("Initializing the model...")
    model = models.resnet18(weights='DEFAULT') #pretrained to make training faster (not necessary > can remove this)
    num_ftrs = model.fc.in_features #get feature number from last layer
    model.fc = nn.Linear(num_ftrs, num_classes)
    print("Model initialized.")

    criterion = nn.CrossEntropyLoss() #loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001) #learning rate

    for epoch in range(num_epochs): #training loop
        print(f"Starting epoch {epoch+1}/{num_epochs}...")
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad() #zero gradients before each batch
            outputs = model(images) #forward pass
            loss = criterion(outputs, labels) #loss calculation
            loss.backward() #back propogation
            optimizer.step() #weight updates
            running_loss += loss.item() * images.size(0)
            print(f"Batch {i}/{len(train_loader)} processed with loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset) #calculate, log average loss for epoch
        print(f'Epoch {epoch+1}/{num_epochs} completed. Training Loss: {epoch_loss:.4f}')

        model.eval() #validation loop
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad(): #no gradient computation for validation
            for i, (images, labels) in enumerate(val_loader, 1):
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1) #get predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print(f"Validation Batch {i}/{len(val_loader)} processed with loss: {loss.item():.4f}")

        val_loss = val_loss / len(val_loader.dataset) #validation loss/accuracy
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

    torch.save(model.state_dict(), 'model.pth') #save model weights
    print('Model training completed and saved to model.pth')

if __name__ == "__main__":
    # load data from .npy files
    images, labels = load_data_from_files('images.npy', 'labels.npy')
    
    # encode labels
    labels_encoded, label_encoder = encode_labels(labels)
    
    # determine the number of unique classes
    num_classes = len(np.unique(labels_encoded))
    
    # split data according to set size
    train_size = 12800
    val_size = 6400
    test_size = 6400
    #array splicing respective to above
    X_train = images[:train_size]
    y_train = labels_encoded[:train_size]
    X_val = images[train_size:train_size + val_size]
    y_val = labels_encoded[train_size:train_size + val_size]
    X_test = images[train_size + val_size:train_size + val_size + test_size]
    y_test = labels_encoded[train_size + val_size:train_size + val_size + test_size]

    # define transformations for input data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),     # resize image to 224x224
        transforms.ToTensor(),             # convert PIL Image to tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize with imagenet values
    ])

    # create datasets for training/validation
    print("Creating datasets...")
    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    val_dataset = CustomDataset(X_val, y_val, transform=transform)

    # create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Batch size 32
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)    # Batch size 32

    # start model training
    print("Starting model training...")
    train_model(train_loader, val_loader, num_epochs=10, num_classes=num_classes)
