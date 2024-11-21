import subprocess

def test_pipeline():
    # Train the model
    subprocess.run(["python", "train.py"], check=True)

    # Validate the model
    subprocess.run(["python", "validate.py"], check=True)

    # Deploy the model
    subprocess.run(["python", "deploy.py"], check=True)

    print("All tests passed.")

if __name__ == "__main__":
    test_pipeline()
