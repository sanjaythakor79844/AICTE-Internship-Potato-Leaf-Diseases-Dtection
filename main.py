import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Potato Disease Classification", layout="centered", initial_sidebar_state="expanded")

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.h5")  # Replace with the actual model path

model = load_model()

# Define class labels
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Define prediction function
def predict_disease(image):
    img = Image.open(image).convert('RGB')  # Convert to RGB
    img = img.resize((128, 128))  # Resize for the model
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    return class_names[predicted_class], confidence

# UI Design
st.markdown("<h1 style='text-align: center; color:  #FFFFFF;'>Potato Disease Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #FFFFFF;'>Upload an image of a potato leaf to predict if it has Early Blight, Late Blight, or is Healthy.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# If file uploaded
if uploaded_file:
    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Prediction button
    if st.button("Predict"):
        with st.spinner("Analyzing the image..."):
            result, confidence = predict_disease(uploaded_file)

        # Display results
        st.success(f"Prediction: **{result}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

# Footer styling
st.markdown("""
    <style>
        .reportview-container {
            background: #264653;
        }
        .css-1d391kg {
            color: white;
        }
        footer {
            color: #E9C46A;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)
