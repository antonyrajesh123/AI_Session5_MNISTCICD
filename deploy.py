import torch
import time
from train import DNN

def deploy_model():
    model = DNN()
    model.load_state_dict(torch.load("model.pth"))

    # Save with timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"model_{timestamp}.pth"
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    deploy_model()
