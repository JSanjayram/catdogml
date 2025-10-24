import streamlit as st
from cat_dog_model import CatDogClassifier
import tempfile
import os
from PIL import Image
import requests
from io import BytesIO

# Initialize the classifier
@st.cache_resource
def load_model():
    classifier = CatDogClassifier(confidence_threshold=0.9)
    classifier.build_model()
    return classifier

def process_image(image, classifier):
    """Process image and return prediction results"""
    with st.spinner('Analyzing image...'):
        result = classifier.predict_image(image)
    return result

def display_results(result):
    """Display prediction results"""
    st.subheader("Results:")
    if result['status'] == 'confident':
        st.success(f"**Prediction: {result['prediction']}**")
        st.info(f"Confidence: {result['confidence']:.2%}")
    else:
        st.warning("**Uncertain Classification**")
        st.error(result['message'])
        st.info(f"Current confidence: {result['confidence']:.2%}")

def main():
    st.title("🐱🐶 Cat vs Dog Classifier")
    st.write("Upload an image or provide URL to classify if it's a cat or dog (90% confidence required)")
    
    classifier = load_model()
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Upload File", "Image URL"])
    
    image = None
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
    
    else:  # Image URL
        url = st.text_input("Enter image URL:")
        if url:
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                st.image(image, caption='Image from URL', use_column_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    if image is not None:
        result = process_image(image, classifier)
        display_results(result)

if __name__ == "__main__":
    main()