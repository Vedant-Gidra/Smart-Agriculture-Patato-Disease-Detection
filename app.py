from flask import Flask, request, jsonify , render_template
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingImports]
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

IMG_SIZE = (224, 224)
CHANNELS = 3

def split_image(image):
    upper_half = image[:, :IMG_SIZE[0] // 2, :, :]
    lower_half = image[:, IMG_SIZE[0] // 2:, :, :]
    return upper_half, lower_half

def flip_lower_half(lower_half):
    return tf.image.flip_left_right(lower_half)

model = load_model(
    "Model/potato_leaf_model.keras",
    custom_objects={
        "split_image": split_image,
        "flip_lower_half": flip_lower_half
    }
)

CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]

@app.route("/preprocess", methods=["POST"])
def preprocess_image():
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    np.save("temp_image.npy", image_array)
    return jsonify({"message": "Image preprocessed successfully"})


@app.route("/predict", methods=["GET"])
def predict():
    image_array = np.load("temp_image.npy")
    predictions = model.predict(image_array)[0] 

    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    probabilities = {CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASS_NAMES))}

    return jsonify({
        "predicted_class": predicted_class,
        "class_probabilities": probabilities,
    })

@app.route("/")
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
