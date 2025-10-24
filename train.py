from model import CatDogClassifier
import os

def main():
    # Initialize classifier
    classifier = CatDogClassifier()
    
    # Train the model (requires data/train folder with cat and dog subfolders)
    if os.path.exists('data/train'):
        print("Training model...")
        classifier.train('data/train', epochs=10)
        print("Training completed! Model saved to models/cat_dog_model.keras")
    else:
        print("Please create data/train folder with 'cat' and 'dog' subfolders containing images")

if __name__ == "__main__":
    main()