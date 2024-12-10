from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('personalized_healthcare_model.pkl')

# Endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON data
    # Assume the input is a JSON object with user data, e.g., { "age": 30, "bmi": 22, ... }
    user_input = np.array([data['features']])  # Convert to numpy array for prediction
    prediction = model.predict(user_input)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
