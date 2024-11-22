import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SmallDNN(nn.Module):
    def __init__(self):
        super(SmallDNN, self).__init__()
        # Convolutional Layer: Input (1, 28, 28), Output (16, 14, 14)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2

        # Convolutional Layer: Input (16, 14, 14), Output (32, 7, 7)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2

        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Hidden layer
        self.fc2 = nn.Linear(128, 10)         # Output layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Model Summary
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model = SmallDNN()
print(f"Total Parameters: {count_parameters(model)}")

# Training function
# Training script
def train_model():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 1

    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = SmallDNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    total, correct = 0, 0
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    train_model()
