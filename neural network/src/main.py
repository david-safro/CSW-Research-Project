import torch
from torch import nn
import pickle
from data import data_preprocessing
from constants import *
import json

def json_to_input(json_data):
    try:
        data = json.loads(json_data)
                input_data = [
            data.get("age", 0),
            data.get("sex", 0),
            data.get("cp", 0),
            data.get("trestbps", 0),
            data.get("chol", 0),
            data.get("fbs", 0),
            data.get("restecg", 0),
            data.get("thalach", 0),
            data.get("exang", 0),
            data.get("oldpeak", 0),
            data.get("slope", 0)
        ]
        
        return input_data
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", str(e))
        return None

def make_prediction(model, scaler, input_data):
    try:
        input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
        
        input_data = torch.tensor(scaler.transform(input_data), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            prediction_logits = model(input_data).squeeze()
            prediction = torch.round(torch.sigmoid(prediction_logits)).cpu().item()
        
        return prediction
    except Exception as e:
        print("Error making prediction:", str(e))
        return None

if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    load_model_weights(model, 'model_weights.pkl')
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    
    # Example JSON input (modify this as needed)
    example_json_input = '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0}'
        input_data = json_to_input(example_json_input)
    
    if input_data:
        prediction = make_prediction(model, scaler, input_data)
        
        if prediction is not None:
            if prediction == 1:
                return True
            else:
                return False
        else:
            raise Exception("Error couldn't predict")
