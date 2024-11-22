import torch
from torchvision import datasets, transforms
from train import SmallDNN

def validate_model():
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.util.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SmallDNN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    assert accuracy >= 95, "Validation Accuracy is less than 95%!"
    print("Model validation passed.")
