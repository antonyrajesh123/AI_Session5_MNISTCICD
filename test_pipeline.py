import subprocess

def test_pipeline():
    print("Starting training...")
    subprocess.run(["python", "train.py"], check=True)

    print("Starting validation...")
    subprocess.run(["python", "validate.py"], check=True)

    print("Starting deployment...")
    subprocess.run(["python", "deploy.py"], check=True)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    test_pipeline()
