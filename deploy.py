import torch
import time
from train import SmallDNN

def deploy_model():
    model = SmallDNN()
    model.load_state_dict(torch.load("model.pth"))

    # Save the model with a timestamp suffix
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"model_{timestamp}.pth"
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    deploy_model()
