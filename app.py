import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import requests
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    if os.path.exists('models/cat_dog_model.keras'):
        return tf.keras.models.load_model('models/cat_dog_model.keras')
    return None

def preprocess_image(image):
    """Preprocess uploaded image"""
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32')
    return np.expand_dims(img, axis=0)

def main():
    st.title("🐱🐶 Cat vs Dog Classifier")
    st.markdown("Upload an image to classify if it's a cat or dog!")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first by running `python setup_and_train.py`")
        return
    
    # Input options
    input_method = st.radio("Choose input method:", ["Upload Image", "Image URL"])
    
    image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of a cat or dog"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    
    else:  # Image URL
        url = st.text_input("Enter image URL:", placeholder="https://example.com/image.jpg")
        if url:
            try:
                with st.spinner("Loading image from URL..."):
                    response = requests.get(url, timeout=10)
                    image = Image.open(BytesIO(response.content))
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    if image is not None:
        # Display image
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.image(image, caption="Image to Classify", use_column_width=True)
        
        # Predict button
        if st.button("🔍 Classify Image", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Preprocess and predict
                    processed_img = preprocess_image(image)
                    prediction = model.predict(processed_img, verbose=0)[0][0]
                    
                    # Display results with confidence threshold
                    confidence = float(prediction)
                    max_confidence = max(confidence, 1-confidence)
                    
                    if max_confidence < 0.6:
                        st.warning(f"⚠️ **UNCERTAIN** - Low confidence ({max_confidence:.1%})")
                        st.info("Try a clearer image with better lighting")
                    elif confidence > 0.5:
                        st.success(f"🐶 **DOG** (Confidence: {confidence:.1%})")
                    else:
                        st.success(f"🐱 **CAT** (Confidence: {1-confidence:.1%})")
                    
                    st.progress(max_confidence)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This AI model uses deep learning to classify images as cats or dogs.")
        
        st.header("📊 Model Info")
        st.write("- **Architecture**: CNN")
        st.write("- **Input Size**: 128x128 pixels")
        st.write("- **Training Data**: 600 images")
        
        st.header("💡 Tips")
        st.write("- Use clear, well-lit images")
        st.write("- Ensure the animal is the main subject")
        st.write("- JPG, JPEG, PNG formats supported")
        st.write("- URLs must be direct image links")

if __name__ == "__main__":
    main()