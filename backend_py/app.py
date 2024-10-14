from flask import Flask, request, jsonify
import numpy as np
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('asddetection_model.pkl')

@app.route('/')
def index():
    return "ASD Detection Model Prediction Service"

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the data from the request
        data = request.json
        
        # Extract individual features
        reaction_time_npc = data.get('reaction_time_npc', 0)
        reaction_time_duck = data.get('reaction_time_duck', 0)
        accuracy = data.get('accuracy', 0)
        impulsivity = data.get('impulsivity', 0)
        task_switch_freq = data.get('task_switch_freq', 0)
        social_vs_hunting = data.get('social_vs_hunting', 0)
        
        # Create the feature array (1 sample, n features)
        features = np.array([[reaction_time_npc, reaction_time_duck, accuracy, impulsivity, task_switch_freq, social_vs_hunting]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Return the result as JSON
        return jsonify({
            'prediction': int(prediction),
            'probability': prediction_proba.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
