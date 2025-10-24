import os
import streamlit as st
from download_data import download_dataset
from model import CatDogClassifier

def auto_train_on_first_run():
    """Auto-train model on first deployment if not exists"""
    if not os.path.exists('models/cat_dog_model.keras'):
        st.info("🚀 First time deployment detected. Training model...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Download data
            status_text.text("📥 Downloading training data...")
            progress_bar.progress(20)
            download_dataset()
            
            # Train model
            status_text.text("🤖 Training AI model...")
            progress_bar.progress(50)
            classifier = CatDogClassifier()
            classifier.train('data/train', epochs=10)
            
            progress_bar.progress(100)
            status_text.text("✅ Model trained successfully!")
            st.success("🎉 Model ready for predictions!")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            return False
    
    return True