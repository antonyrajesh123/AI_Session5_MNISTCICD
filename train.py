import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define the DNN
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Convolutional Layer
        self.fc1 = nn.Linear(16 * 28 * 28, 128)                           # Fully Connected Layer
        self.fc2 = nn.Linear(128, 10)                                    # Output Layer

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply ReLU after convolution
        x = x.view(x.size(0), -1)      # Flatten the tensor
        x = torch.relu(self.fc1(x))   # Apply ReLU after FC layer
        x = self.fc2(x)               # Output layer
        return x

# Training function
def train_model():
    # Download and prepare dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = DNN()
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
