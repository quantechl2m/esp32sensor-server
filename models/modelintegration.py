import numpy as np
import tensorflow as tf
import os
from flask import request, jsonify
from PIL import Image
import io

# Function to preprocess image and convert it into a format compatible with the model
def preprocess_image(image_file, target_size=(256, 256)):
    try:
        # Open the image file with PIL (Pillow)
        img = Image.open(io.BytesIO(image_file.read()))
        img = img.convert("RGB")  # Ensure the image is in RGB format
        img = img.resize(target_size)  # Resize the image to the target size

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Normalize the image array and expand dimensions to match the input shape (batch size, height, width, channels)
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension

        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

# Function to detect disease (generic for all plant diseases)
def detect_disease(model_path, class_labels):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]

    try:
        img_array = preprocess_image(image)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Set tensor input
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    # Get output and return prediction
    output_data = interpreter.get_tensor(output_details[0]["index"])
    predicted_class_index = np.argmax(output_data)
    predicted_class_label = class_labels[predicted_class_index]

    return jsonify({"predicted_label": predicted_class_label, "output": output_data.tolist()})

# Function to detect soil type
def detect_soil(model_path, class_labels):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]

    try:
        img_array = preprocess_image(image)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Set tensor input
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    # Get output and return prediction
    output_data = interpreter.get_tensor(output_details[0]["index"])
    predicted_class_index = np.argmax(output_data)
    predicted_class_label = class_labels[predicted_class_index]

    return jsonify({"predicted_label": predicted_class_label, "output": output_data.tolist()})

# Specific function to detect potato disease
def detect_potato_disease():
    model_path = os.path.join(os.getcwd(), "models", "potato_disease_detector.tflite")
    return detect_disease(model_path=model_path, class_labels=["EARLY_BLIGHT", "HEALTHY", "LATE_BLIGHT", "RANDOM"])

# Specific function to detect soyabean disease
def detect_soyabean_disease():
    model_path = os.path.join(os.getcwd(), "models", "soyabean_disease_detector_1.tflite")
    return detect_disease(
        model_path=model_path,
        class_labels=[
            "Mossaic Virus", 
            "Southern blight", 
            "Sudden Death Syndrone", 
            "Yellow Mosaic", 
            "bacterial_blight",
            "brown_spot",
            "crestamento",
            "ferrugen",
            "powdery_mildew",
            "septoria"
        ]
    )

# Specific function to detect maize disease
def detect_maize_disease():
    model_path = os.path.join(os.getcwd(), "models", "maize_disease_detector.tflite")
    return detect_disease(model_path=model_path, class_labels=["BLIGHT", "COMMON_RUST", "GRAY_LEAF_SPOT", "HEALTHY", "RANDOM"])

# Specific function to detect wheat disease
def detect_wheat_disease():
    model_path = os.path.join(os.getcwd(), "models", "wheat_disease_detector.tflite")
    return detect_disease(model_path=model_path, class_labels=["CROWN_AND_ROOT_ROT", "HEALTHY", "LEAF_RUST", "RANDOM", "WHEAT_LOOSE_SMUT"])

# Function to detect soil type (with model for soil detection)
def detect_the_soil():
    model_path = os.path.join(os.getcwd(), "models", "soil_detector.tflite")
    return detect_soil(model_path=model_path, class_labels=["ALLUVIAL_SOIL", "BLACK_SOIL", "RANDOM", "RED_SOIL"])

