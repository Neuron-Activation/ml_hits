from flask import Flask, request
from model import My_Classifier_Model
import argparse

app = Flask(__name__)
model = My_Classifier_Model()

@app.route('/')
def home():
    return 'Hello, welcome to my Flask app!'

@app.route('/train', methods=['POST'])
def train():
    dataset = request.json['dataset']
    model.train(dataset)
    return 'Model trained successfully'

@app.route('/predict', methods=['POST'])
def predict():
    dataset = request.json['dataset']
    prediction = model.make_prediction(dataset)
    return {'prediction': prediction}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)