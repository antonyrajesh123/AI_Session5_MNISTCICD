import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the model
class SmallDNN(nn.Module):
    def __init__(self):
        super(SmallDNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)  # Fewer filters
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)  # Reduce filters
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 3 * 3, 32)  # Further reduce neurons
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.0005  # Lower learning rate
    num_epochs = 1

    # Data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = SmallDNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Validate parameter count
    param_count = count_parameters(model)
    print(f"Model Parameters: {param_count}")
    assert param_count < 25000, f"Model has too many parameters: {param_count}"

    # Training loop
    total, correct = 0, 0
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        training_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {training_accuracy:.2f}%")
    
    assert training_accuracy >= 80, f"Training accuracy is below 95%: {training_accuracy:.2f}%"
    print("Model passed all tests!")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_test_model()