from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# loading model
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('notebook and dataset/scaler.pkl','rb'))

# flask app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['feature']

    # split + clean spaces
    features = [float(x.strip()) for x in features.split(',')]

    print("Features length:", len(features))  # debug

    np_features = np.array(features).reshape(1, -1)

    # 🔥 APPLY SCALING (MOST IMPORTANT FIX)
    np_features = scaler.transform(np_features)

    pred = model.predict(np_features)

    print("Prediction:", pred)  # debug

    message = ['Cancrouse' if pred[0] == 1 else 'Not Cancrouse']

    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)