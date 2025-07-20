import os
import numpy as np
import tensorflow as tf
import requests
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Model path and Google Drive ID
MODEL_PATH = "dog_skin397.h5"
GOOGLE_DRIVE_FILE_ID = "12uG--fQ5-rhQ1kzmjeUJo3VJnFlVMYOS"

# Function to download model
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
        response = requests.get(url)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("Download complete.")
        else:
            raise Exception("Failed to download model.")

# Download and load the model
download_model()
model = tf.keras.models.load_model(MODEL_PATH)

# Disease classes
class_labels = ["Flea allergy", "Hotspot", "Mange"]

# Remedies
remedies = {
    "Flea allergy": "Flea Allergy Dermatitis (FAD) occurs due to flea saliva...",
    "Hotspot": "Hotspots develop from allergies or moisture buildup...",
    "Mange": "Mange, caused by mites, leads to intense itching and hair loss...",
}

# Uploads folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Process image
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]
            remedy = remedies.get(predicted_class, "Consult a veterinarian.")

            return render_template(
                "index.html", result=predicted_class, remedy=remedy, img_path=filepath
            )

    return render_template("index.html", result=None, remedy=None)

# Run app
if __name__ == "__main__":
    app.run(debug=True)
