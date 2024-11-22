import torch
import time
from train_and_test import SmallDNN

def deploy_model():
    model = SmallDNN()
    print("1")
    model.load_state_dict(torch.load("model.pth"))
    print("2")
    # Save the model with a timestamp suffix
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("3")
    filename = f"model_{timestamp}.pth"
    print("4")
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    deploy_model()
