from model import CatDogClassifier
import os
from pathlib import Path

def test_with_training_images():
    """Test model with images from training data"""
    classifier = CatDogClassifier()
    
    if not os.path.exists('models/cat_dog_model.keras'):
        print("No trained model found. Run setup_and_train.py first.")
        return
    
    classifier.load_model()
    
    # Test with a cat image
    cat_images = list(Path('data/train/cat').glob('*.jpg'))
    if cat_images:
        result = classifier.predict(cat_images[0])
        print(f"Cat image prediction: {result}")
    
    # Test with a dog image  
    dog_images = list(Path('data/train/dog').glob('*.jpg'))
    if dog_images:
        result = classifier.predict(dog_images[0])
        print(f"Dog image prediction: {result}")

if __name__ == "__main__":
    test_with_training_images()