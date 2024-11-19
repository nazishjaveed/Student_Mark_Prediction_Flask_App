from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('marks_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    study_hours = float(request.form['study_hours'])
    attendance = float(request.form['attendance'])
    previous_marks = float(request.form['previous_marks'])

    # Predict the final marks
    prediction = model.predict(np.array([[study_hours, attendance, previous_marks]]))
    
    return render_template('result.html', predicted_marks=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
