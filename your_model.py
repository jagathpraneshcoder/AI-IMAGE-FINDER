import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def predict_ai_generated(model_path, image_path):
    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0  # Normalize pixel values to be between 0 and 1

    # Make a prediction using the model
    prediction = model.predict(img_array)

    # Print the result
    if prediction[0][0] < 0.1:
        print(f"The image '{image_path}' is AI-generated.")
    else:
        print(f"The image '{image_path}' is likely not AI-generated.")

# Example usage
model_path = "E:/your_model.h5"  # Replace with your trained model path
image_path = "C:/Users/Jagathpranesh/Downloads/photo_2023-11-18_22-11-00.jpg"
# Replace with the path to the image you want to test

predict_ai_generated(model_path, image_path)
