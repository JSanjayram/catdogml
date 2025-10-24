from model import CatDogClassifier
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        return
    
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    # Load trained model and predict
    classifier = CatDogClassifier()
    
    if os.path.exists('models/cat_dog_model.keras'):
        classifier.load_model()
        result = classifier.predict(img_path)
        print(f"Prediction: {result}")
    else:
        print("No trained model found. Please run train.py first.")

if __name__ == "__main__":
    main()