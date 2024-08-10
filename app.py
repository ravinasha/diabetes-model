from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    try:
        # Retrieve form data
        Pregnancies = int(request.form.get('Pregnancies'))
        Glucose = int(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = int(request.form.get('SkinThickness'))
        Insulin = int(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))  # Ensure this is float if needed
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = int(request.form.get('Age'))

        # Create feature array for prediction
        features = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)

        # Make prediction
        result = model.predict(features)

        # Determine result label based on prediction
        if result[0] == 1:
            result_label = 'Diabetes'
        else:
            result_label = 'Not Diabetes'

    except Exception as e:
        return f"Error processing request: {e}"

    return render_template('index.html', result=result_label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
