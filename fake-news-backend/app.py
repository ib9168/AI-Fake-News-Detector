#This file loads  fine-tuned model and exposes it to the internet.
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import os

app = Flask(__name__)
# Enable CORS to allow React (port 5173/3000) to talk to Flask (port 5000)
CORS(app)

# Configuration 
MODEL_PATH = './saved_model'  # Path where train_model.py saved the files
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Loading model from {MODEL_PATH}...")
print(f"Using device: {device}")

#  Load Model & Tokenizer 
try:
    # Loading the tokenizer and model saved earlier
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()  # Set model to evaluation mode
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Did you run 'train_model.py' first?")
    model = None

#  Helper Function 
def get_prediction(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Applying softmax to get probabilities (confidence scores)
    probs = F.softmax(outputs.logits, dim=1)
    
    # Getting the highest probability class
    # ASSUMPTION: In  CSV, 0 = Fake, 1 = Real. Adjusting if data is opposite.
    prediction_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][prediction_idx].item()
    
    labels = {0: "Fake", 1: "Real"}
    
    return {
        "label": labels[prediction_idx],
        "confidence": round(confidence * 100, 2),
        "raw_scores": {
            "fake_prob": round(probs[0][0].item(), 4),
            "real_prob": round(probs[0][1].item(), 4)
        }
    }

#  API Routes 

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = get_prediction(text)
        return jsonify({
            "text_snippet": text[:50] + "...",
            "prediction": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)