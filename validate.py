import torch
from torchvision import datasets, transforms
from train import DNN

def validate_model():
    model = DNN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Check parameter count
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 100000, f"Model has too many parameters: {param_count}"

    # Check input-output compatibility
    dummy_input = torch.randn(1, 1, 28, 28)  # Batch of 1, 1 channel, 28x28 image
    output = model(dummy_input)
    assert output.shape == (1, 10), f"Invalid output shape: {output.shape}"

    # Check accuracy
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    assert accuracy > 0.8, f"Accuracy too low: {accuracy:.2f}"

    print("Model validation passed.")

if __name__ == "__main__":
    validate_model()
