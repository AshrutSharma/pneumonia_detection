from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow requests from React frontend

# Load your model once when the server starts
MODEL_PATH = 'resnet152_pneumonia_baseline.keras'

print("Building model architecture...")
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras import layers, models

IMG_SIZE = 224
base_model = ResNet152V2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

print("Loading trained weights...")
try:
    # Try loading the full model first
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"Could not load full model, trying weights only: {e}")
    # If that fails, load just the weights
    model.load_weights(MODEL_PATH)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("✅ Model weights loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        # Read image file
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Preprocess image (same as training)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        # Make prediction
        prediction = model.predict(img)[0][0]
        
        # Return result
        result = {
            'prediction': float(prediction),
            'has_pneumonia': bool(prediction > 0.5),
            'confidence': float(max(prediction, 1 - prediction) * 100)
        }
        
        print(f"Prediction: {result}")
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})

if __name__ == '__main__':
    print("\n🚀 Server starting on http://0.0.0.0:5000")
    print("📝 Endpoints:")
    print("   - POST /predict - Make predictions")
    print("   - GET  /health  - Check server status")
    app.run(host='0.0.0.0', debug=True, port=5000)