# Description: This file contains the code for integrating the TFLite model with the Flask API
import numpy as np
import tensorflow as tf
import os
from flask import request, jsonify


def detect_disease(model_path, class_labels):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if "image" not in request.files:
        return jsonify({"error": "No image provided"})

    image = request.files["image"]

    img_array = tf.image.decode_image(image.read())
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = tf.expand_dims(img_array, 0) / 255.0

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    predicted_class_index = np.argmax(output_data)

    predicted_class_label = class_labels[predicted_class_index]

    return jsonify(
        {"predicted_label": predicted_class_label, "output": output_data.tolist()}
    )

def detect_soil(model_path, class_labels):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if "image" not in request.files:
        return jsonify({"error": "No image provided"})

    image = request.files["image"]

    img_array = tf.image.decode_image(image.read())
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = tf.expand_dims(img_array, 0) / 255.0

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    predicted_class_index = np.argmax(output_data)

    predicted_class_label = class_labels[predicted_class_index]

    return jsonify(
        {"predicted_label": predicted_class_label, "output": output_data.tolist()}
    )



# Function to detect potato disease
def detect_potato_disease():
    # Load the TFLite model
    model_path = os.path.join(os.getcwd(), "models", "potato_disease_detector.tflite")
    return detect_disease(model_path=model_path, class_labels=["EARLY_BLIGHT", "HEALTHY", "LATE_BLIGHT","RANDOM"])
    

def detect_soyabean_disease():
#     # Load the TFLite model
     model_path = os.path.join(os.getcwd(), "models", "soyabean_disease_detector.tflite")
     return detect_disease(model_path=model_path , class_labels=["CATERPILLAR","DIABROTICA_SPECIOSA", "HEALTHY", "RANDOM"])


def detect_maize_disease():
#     # Load the TFLite model
     model_path = os.path.join(os.getcwd(), "models", "maize_disease_detector.tflite")
     return detect_disease(model_path=model_path , class_labels=["BLIGHT","COMMON_RUST", "GRAY_LEAF_SPOT", "HEALTHY", "RANDOM"])

def detect_wheat_disease():
#     # Load the TFLite model
     model_path = os.path.join(os.getcwd(), "models", "wheat_disease_detector.tflite")
     return detect_disease(model_path=model_path , class_labels=["CROWN_AND_ROOT_ROT", "HEALTHY", "LEAF_RUST","RANDOM", "WHEAT_LOOSE_SMUT"])

def detect_the_soil():
#     # Load the TFLite model
     model_path = os.path.join(os.getcwd(), "models", "soil_detector.tflite")
     return detect_soil(model_path=model_path , class_labels=["ALLUVIAL_SOIL","BLACK_SOIL", "RANDOM", "RED_SOIL"])

     
     