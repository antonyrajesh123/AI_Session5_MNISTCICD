import torch
from torchvision import datasets, transforms
from train import SmallDNN

def test_deployment():
    # Load the deployed model (using the most recent model saved with timestamp)
    model_files = sorted(glob.glob("model_*.pth"), reverse=True)  # Sort models by date
    latest_model = model_files[0]  # Get the most recent model

    model = SmallDNN()
    model.load_state_dict(torch.load(latest_model))
    model.eval()

    # Run inference on a sample image from MNIST test set
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Pick a random image for testing
    images, labels = next(iter(test_loader))
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print(f"Predicted: {predicted.item()}, Actual: {labels.item()}")
    assert predicted.item() == labels.item(), "Model prediction mismatch"

    print("Deployment test passed.")

if __name__ == "__main__":
    test_deployment()
