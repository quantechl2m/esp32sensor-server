from flask import Flask
from models.modelintegration import detect_potato_disease, detect_soyabean_disease, detect_maize_disease,detect_the_soil,detect_wheat_disease
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():
    return "Welcome to esp32sensor server!"


@app.route("/disease-detection/potato", methods=["POST"])
def potato_disease_controller():
    return detect_potato_disease()


@app.route("/disease-detection/maize", methods=["POST"])
def maize_disease_controller():
    return detect_maize_disease()


@app.route("/disease-detection/soyabean", methods=["POST"])
def soyabean_disease_controller():
     return detect_soyabean_disease()

@app.route("/type-detection/soil", methods=["POST"])
def soil_detection_controller():
     return detect_the_soil()

@app.route("/disease-detection/wheat", methods=["POST"])
def wheat_disease_controller():
     return detect_wheat_disease()


http_server = WSGIServer(("0.0.0.0", 8000), app)
http_server.serve_forever()
