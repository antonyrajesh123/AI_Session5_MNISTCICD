import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the model
class SmallDNN(nn.Module):
    def __init__(self):
        super(SmallDNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Count parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Train the model
def train_and_test_model():
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

    # Validate parameter count
    param_count = count_parameters(model)
    print(f"Model Parameters: {param_count}")
    assert param_count < 25000, f"Model has too many parameters: {param_count}"

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

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        training_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {training_accuracy:.2f}%")
    
    # Validate accuracy threshold
    assert training_accuracy >= 95, f"Training accuracy is below 95%: {training_accuracy:.2f}%"
    print("Model passed all tests!")

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    train_and_test_model()
