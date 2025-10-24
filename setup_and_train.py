from download_data import download_dataset
from model import CatDogClassifier
import os

def main():
    # Download and setup dataset
    if not os.path.exists('data/train/cat') or len(os.listdir('data/train/cat')) == 0:
        print("Setting up dataset...")
        download_dataset()
    
    # Train the model
    print("Starting training...")
    classifier = CatDogClassifier()
    classifier.train('data/train', epochs=20)
    print("Training completed! Model saved.")

if __name__ == "__main__":
    main()