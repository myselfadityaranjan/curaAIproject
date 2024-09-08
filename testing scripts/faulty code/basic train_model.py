import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from load_data import load_dicom_images

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 64 * 64, 2)  # Adjust based on your image size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)  # Flatten the output
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    data_dir = "../CURA AI training images/"
    images, labels = load_dicom_images (data_dir)

    # Convert to PyTorch tensors
    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # Normalize images if needed
    transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    images = transform(images)

    # Create a dataset and data loader
    dataset = TensorDataset(images, labels)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(10):  # Number of epochs
        for i, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch + 1}/10], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), "../models/brain_cancer_model.pth")