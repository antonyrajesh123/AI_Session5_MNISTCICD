import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define the DNN
class SmallDNN(nn.Module):
    def __init__(self):
        super(SmallDNN, self).__init__()
        # Convolutional Layer: Input (1, 28, 28), Output (8, 14, 14)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        # Fully Connected Layers
        self.fc1 = nn.Linear(8 * 14 * 14, 32)  # Reduce hidden layer size
        self.fc2 = nn.Linear(32, 10)          # Output layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Convolution + ReLU
        x = self.pool(x)               # Pooling
        x = x.view(x.size(0), -1)      # Flatten
        x = torch.relu(self.fc1(x))   # Fully connected + ReLU
        x = self.fc2(x)               # Output layer
        return x

# Training function
def train_model():
    # Download and prepare dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SmallDNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 1 epoch
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save model
    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    train_model()
