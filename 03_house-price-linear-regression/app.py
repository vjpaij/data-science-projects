from flask import Flask, request, jsonify
import pickle
import pandas as pd
import json

app = Flask(__name__)

# Load artifacts
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
        
    with open('feature_names.json', 'r') as f:
        feature_names = json.load(f)

except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please make sure model.pkl, preprocessor.pkl, and feature_names.json exist")
    exit(1)

@app.route('/')
def home():
    return "House Price Prediction API - Send POST request to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Convert to DataFrame with correct feature order
        input_data = pd.DataFrame({k: [v] for k, v in data.items()})
        
        # Validate input features
        missing_features = set(feature_names) - set(input_data.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}',
                'required_features': feature_names
            }), 400
            
        # Preprocess and predict
        processed = preprocessor.transform(input_data)
        prediction = model.predict(processed)
        
        return jsonify({
            'prediction': float(prediction[0]),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Invalid input format'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)