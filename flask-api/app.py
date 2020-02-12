from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import flask

app = Flask(__name__)

# Load Model
model = load_model('../model.hdf5')

# Variables
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif']
categories = ['ao', 'bus', 'car', 'ete', 'van', 'vuc']

# filter extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# App Flask
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if flask.request.method == 'GET':
        return "Prediction Page API"

    if flask.request.method == 'POST':
        try:
            requested_img = request.files['file']

            if requested_img and allowed_file(requested_img.filename):

                #load img request
                image = np.asarray(bytearray(requested_img.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)
                image = image / np.mean(image)

                # Create array dimesion for model(vgg19 [batch_size, weight, height, rgb]
                x = image.reshape((1,) + image.shape)

                #prediction
                prediction = model.predict(x)

                #return result
                return jsonify({
                    "Result": categories[np.argmax(prediction)],
                    "Sucess": "True"
                })
        except:
            return jsonify(
                {
                    "Sucess" : "False"
                })





if __name__ == '__main__':
    app.run(debug=True)
