from model import CatDogClassifier
import os

def retrain_model():
    """Retrain with better parameters for accuracy"""
    if not os.path.exists('data/train/cat'):
        print("No training data found. Run setup_and_train.py first.")
        return
    
    print("Retraining model with better parameters...")
    classifier = CatDogClassifier()
    
    # Train with more epochs and validation
    classifier.train('data/train', epochs=30)
    print("Better model trained and saved!")

if __name__ == "__main__":
    retrain_model()