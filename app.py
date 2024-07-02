import os
import streamlit as st
import numpy as np
import json
from PIL import Image
from tensorflow import keras
from dotenv import load_dotenv
import google.generativeai as gen_ai

load_dotenv()

# Load the trained model
model = keras.models.load_model("detection_model_mobilenetv2.h5")

# Load class indices from JSON file
class_indices = json.load(open("class_indices.json"))

# Set up Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=GOOGLE_API_KEY)
gemini_model = gen_ai.GenerativeModel('gemini-pro')

# Function to preprocess image
def preprocess_image(image):
    image_rgb = image.convert("RGB")
    resized_image = image_rgb.resize((224, 224))
    normalized_image = np.array(resized_image, dtype=np.float32) / 255.0
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image

# Function to predict diseases
def predict_diseases(model, image, class_indices):
    preprocessed_img = preprocess_image(image)
    probs = model.predict(preprocessed_img)[0]
    pred_class_prob = np.argmax(probs)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name,probs[pred_class_prob]

st.title("Plants Diseases Classifier with AI")
st.subheader("Made by Dhiraj")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        predicted_disease,probability = predict_diseases(model, image, class_indices)
        st.success(f'Predicted Disease: {predicted_disease} \n\n\n Probability: {probability:.2f}')
        if "Healthy" not in predicted_disease:
        # Ask Gemini for information
            with st.spinner("Generating Information"):
                gemini_prompt = f"Tell me about {predicted_disease} disease, its nepali name, its cause, and prevention methods."
                gemini_response = gemini_model.generate_content(gemini_prompt)

        # Display response
                st.subheader("AI Response:")
                st.write(gemini_response.parts[0].text)