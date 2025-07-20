import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "dog_skin397.h5"
model = tf.keras.models.load_model(MODEL_PATH)


# Class labels (Update these based on your dataset)
class_labels = ["Flea allergy", "Hotspot", "Mange"]

# Remedies for each disease
remedies = {
    "Flea allergy": "Flea Allergy Dermatitis (FAD) occurs due to flea saliva, causing itching, hair loss, and scabs. Home remedies include apple cider vinegar spray, aloe vera, coconut oil, and oatmeal baths. Treatment involves flea control medications, antihistamines, medicated shampoos, and antibiotics if needed. Prevent fleas with regular grooming, cleaning, and flea preventatives.",
    "Hotspot": "Hotspots develop from allergies or moisture buildup, presenting as inflamed, oozing sores. Chamomile tea compress, betadine, and aloe vera provide relief, while treatment includes trimming fur, antiseptics, antibiotics, and E-collars to prevent licking. Keep the skin dry and address allergies to prevent hotspots",
    "Mange": "Mange, caused by mites, leads to intense itching and hair loss. Remedies like neem oil, coconut oil, and yogurt help, but vet-prescribed ivermectin, medicated shampoos, and antibiotics may be necessary. Prevent mange with hygiene and immune-boosting diets.",
}

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files["file"]
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Preprocess the image
            img = load_img(
                filepath, target_size=(224, 224)
            )  # Resize to model input size
            img_array = img_to_array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Make prediction
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]

            # Get remedy based on the predicted disease
            remedy = remedies.get(
                predicted_class, "Consult a veterinarian for diagnosis and treatment."
            )

            return render_template(
                "index.html", result=predicted_class, remedy=remedy, img_path=filepath
            )

    return render_template("index.html", result=None, remedy=None)


if __name__ == "__main__":
    app.run(debug=True)
