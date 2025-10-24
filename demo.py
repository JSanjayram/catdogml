from model import CatDogClassifier
import matplotlib.pyplot as plt
import numpy as np

def demo_with_sample():
    """Demo function that creates a simple test without real images"""
    classifier = CatDogClassifier()
    
    # Show model summary
    print("Model Architecture:")
    classifier.model.summary()
    
    # Create dummy data for demonstration
    dummy_data = np.random.random((10, 128, 128, 3))
    dummy_labels = np.random.randint(0, 2, (10, 1))
    
    print("\nTraining on dummy data (for demonstration)...")
    classifier.model.fit(dummy_data, dummy_labels, epochs=1, verbose=1)
    
    # Test prediction on dummy image
    test_img = np.random.random((1, 128, 128, 3))
    prediction = classifier.model.predict(test_img)[0][0]
    result = "Dog" if prediction > 0.5 else "Cat"
    
    print(f"\nDummy prediction: {result} (confidence: {prediction:.2f})")
    print("\nTo use with real images:")
    print("1. Add cat images to data/train/cat/")
    print("2. Add dog images to data/train/dog/")
    print("3. Run: python train.py")
    print("4. Run: python predict.py your_image.jpg")

if __name__ == "__main__":
    demo_with_sample()