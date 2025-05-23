import requests
import json

# Load feature names (save this from your training script)
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Prepare input data in ONE-HOT ENCODED format
input_data = {
    "area": 1500,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "parking": 1,
    # Categorical features must be one-hot encoded
    "mainroad_0": 0,
    "mainroad_1": 1,
    "guestroom_0": 1,
    "guestroom_1": 0,
    "basement_0": 0,
    "basement_1": 1,
    "hotwaterheating_0": 1,
    "hotwaterheating_1": 0,
    "airconditioning_0": 0,
    "airconditioning_1": 1,
    "prefarea_0": 0,
    "prefarea_1": 1,
    "furnishingstatus_furnished": 1,
    "furnishingstatus_semi-furnished": 0,
    "furnishingstatus_unfurnished": 0
}

# Verify we're using all expected features
missing = set(feature_names) - set(input_data.keys())
assert not missing, f"Missing features: {missing}"

response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
print(response.json())