from flask import Flask, request, jsonify
import joblib
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model
model = joblib.load('iris_svc.sav')

# Load the saved scaler object
filename = 'scaler.pickle'
scaler = pickle.load(open(filename, 'rb'))

@app.route('/predict',methods=['POST'])
def predict():
    sepallength = request.form.get('sepallength')
    sepalwidth = request.form.get('sepalwidth')
    petallength = request.form.get('petallength')
    petalwidth = request.form.get('petalwidth')

    new_data = np.array([[sepallength, sepalwidth, petallength, petalwidth]])

    new_data = scaler.transform(new_data)

    prediction = model.predict(new_data)

    species_dict = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    result = species_dict[prediction[0]]

    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)

    
    
