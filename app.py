# app.py
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)

# Set the path where uploaded images will be stored
UPLOAD_FOLDER = 'E:/uploadfolder'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allow only certain file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    # Return the result
    if prediction[0][0] < 0.1:
        return f"The image '{image_path}' is AI-generated."
    else:
        return f"The image '{image_path}' is likely not AI-generated."

@app.route('/')
def index():
    return render_template('upload.html')  # Assuming you have an upload.html template

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return render_template('upload.html', error="No file part")

    file = request.files['image']

    # If the user submits an empty part without selecting a file, the browser will send an empty file without a filename
    if file.filename == '':
        return render_template('upload.html', error="No selected file")

    # If the file is allowed, save it to the uploads folder
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Call your prediction function with the uploaded file
        result = predict_ai_generated("E:/html/your_model.h5", file_path)

        # Pass the result to the template
        return render_template('result.html', filename=filename, result=result)

if __name__ == '__main__':
    app.run(debug=True)
