import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

# print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/covid_attn_weights_best_vgg16.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((150, 150))

    # Preprocessing the image
    x = image.img_to_array(img)
    #x = np.true_divide(x, 255.0)
    #x = x/255.
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')
    #print(x.shape)
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        #print(request.json)
        # Get the image from post request
        img = base64_to_pil(request.json)
        #print(img)
        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)
        print(preds)
        pred_class = np.argmax(preds)
        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        #result = str(pred_class[0][0][1])               # Convert to string
        print(pred_class)
        if pred_class == 0:
            result = 'COVID-19'
        elif pred_class == 1:
            result = 'NORMAL'
        elif pred_class == 2:
            result = 'NEUMONÍA'

        result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
