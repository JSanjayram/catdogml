import subprocess
import sys
import os

def check_model():
    """Check if model exists, if not, train it"""
    if not os.path.exists('models/cat_dog_model.keras'):
        print("Model not found. Training model first...")
        subprocess.run([sys.executable, 'setup_and_train.py'])
    
def run_streamlit():
    """Run the Streamlit app"""
    check_model()
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])

if __name__ == "__main__":
    run_streamlit()