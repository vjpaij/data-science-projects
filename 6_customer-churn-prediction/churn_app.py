from flask import Flask, request, jsonify
import pandas as pd
import torch
import joblib
import os

churn_app = Flask(__name__)

# Check if files exist
if not os.path.exists("preprocessor.pkl"):
    raise FileNotFoundError("preprocessor.pkl not found. Train the model first.")
if not os.path.exists("model.pt"):
    raise FileNotFoundError("model.pt not found. Train the model first.")

# Load preprocessor and model
try:
    preprocessor = joblib.load("preprocessor.pkl")
    model = torch.load("model.pt", map_location=torch.device("cpu"))
    #model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading files: {str(e)}")

@churn_app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Convert to DataFrame and preprocess
        input_df = pd.DataFrame([data])
        processed_data = preprocessor.transform(input_df)
        
        # Predict
        tensor_data = torch.FloatTensor(processed_data.toarray())
        with torch.no_grad():
            output = model(tensor_data)
            prediction = torch.argmax(output).item()
        
        return jsonify({
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "confidence": torch.max(torch.softmax(output, dim=1)).item()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    churn_app.run(host="0.0.0.0", port=5000)