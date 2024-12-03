import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Define the model
class SmallDNN(nn.Module):
    def __init__(self):
        super(SmallDNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=1)  # Reduced to 8 filters
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # Reduced to 16 filters
        self.fc1 = nn.Linear(16 * 12 * 12, 10)  # Adjusted for output size after conv layers

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 16 * 12 * 12)  # Flatten to match the output size
        x = self.fc1(x)
        return x


def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Train the model
# Validate parameter count
model = SmallDNN()
param_count = count_parameters(model)
print(f"Total Parameters: {param_count}")

# Assert parameter count
assert param_count < 25000, f"Model has too many parameters: {param_count}"

def train_and_test_model():
    # Check if the MNIST dataset is already downloaded
    train_images_path = './data/MNIST/raw/train-images-idx3-ubyte'
    train_labels_path = './data/MNIST/raw/train-labels-idx1-ubyte'
    test_images_path = './data/MNIST/raw/t10k-images-idx3-ubyte'
    test_labels_path = './data/MNIST/raw/t10k-labels-idx1-ubyte'

    if not (os.path.exists(train_images_path) and os.path.exists(train_labels_path) and
            os.path.exists(test_images_path) and os.path.exists(test_labels_path)):
        print("Downloading MNIST dataset...")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    model = SmallDNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(1):  # 1 epoch
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log the batch loss
            if batch_idx % 100 == 0:  # Print every 100 batches
                print(f'Epoch [{epoch + 1}/1], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Save the model after training
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as 'model.pth'.")

    # Evaluate the model
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test accuracy: {accuracy:.2f}%')
    assert accuracy >= 95, f"Training accuracy is below 95%: {accuracy:.2f}%"

    # Print the number of parameters in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')

if __name__ == "__main__":
    train_and_test_model()