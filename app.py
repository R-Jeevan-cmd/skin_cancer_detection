import os
from flask import Flask, request, redirect, render_template, send_from_directory, jsonify, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from web3 import Web3
from eth_account.messages import encode_defunct
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Connect to Ethereum Node (Ganache or Infura)
INFURA_URL = "http://127.0.0.1:7545"  # Use Infura URL for Testnet if needed
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

def verify_signature(address, message, signature):
    """Verifies an Ethereum signed message."""
    message_encoded = encode_defunct(text=message)
    recovered_address = web3.eth.account.recover_message(message_encoded, signature=signature)
    return recovered_address.lower() == address.lower()

# -----------------------------
# SPP custom layer (same as before)
# -----------------------------
def spatial_pyramid_pooling(inputs, levels=[1, 2, 4]):
    shape = inputs.shape
    pool_list = []
    for level in levels:
        pool_size = (int(np.ceil(shape[1] / level)), int(np.ceil(shape[2] / level)))
        x = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_size, padding='same')(inputs)
        pooled = tf.keras.layers.Flatten()(x)
        pool_list.append(pooled)
    return tf.keras.layers.Concatenate()(pool_list)

# -----------------------------
# Load centralized model (with custom_objects)
# -----------------------------
CENTRALIZED_MODEL_PATH = 'models/skin_lesion_model.h5'  # existing model in your project
custom_objects = {'spatial_pyramid_pooling': spatial_pyramid_pooling}

centralized_model = None
try:
    print("Loading centralized model from:", CENTRALIZED_MODEL_PATH)
    centralized_model = load_model(CENTRALIZED_MODEL_PATH, custom_objects=custom_objects)
    print("Centralized model loaded.")
except Exception as e:
    print("Could not load centralized model:", e)
    centralized_model = None

# -----------------------------
# Load Federated model (if present)
# -----------------------------
# federated model saved by federated_simulation.py (prefer .keras, fallback to .h5)
FEDERATED_MODEL_PATH_KERAS = "models/federated_global_model.keras"
FEDERATED_MODEL_PATH_H5 = "models/federated_global_model.h5"

federated_model = None
try:
    if os.path.exists(FEDERATED_MODEL_PATH_KERAS):
        print("Loading federated model (KERAS) from:", FEDERATED_MODEL_PATH_KERAS)
        federated_model = load_model(FEDERATED_MODEL_PATH_KERAS)
    elif os.path.exists(FEDERATED_MODEL_PATH_H5):
        print("Loading federated model (H5) from:", FEDERATED_MODEL_PATH_H5)
        federated_model = load_model(FEDERATED_MODEL_PATH_H5)
    else:
        print("No federated model found at expected paths.")
    if federated_model is not None:
        print("Federated model loaded.")
except Exception as e:
    print("Could not load federated model:", e)
    federated_model = None

# -----------------------------
# Upload folder
# -----------------------------
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    return render_template("login.html")  # Serve the login page

@app.route('/verify-login', methods=['POST'])
def verify_login():
    data = request.json
    user_address = data.get("address")
    message = data.get("message")
    signature = data.get("signature")

    if not user_address or not message or not signature:
        return jsonify({"error": "Missing parameters"}), 400

    is_valid = verify_signature(user_address, message, signature)
    if not is_valid:
        return jsonify({"error": "Invalid signature"}), 401

    return jsonify({"redirect": url_for('detection', address=user_address)})

@app.route('/detection')
def detection():
    address = request.args.get("address", "Unknown")
    return render_template('index.html', user_address=address)

# -----------------------------
# Class labels & disease names
# -----------------------------
SIZE_CENTRAL = 64   # your centralized model input size
SIZE_FEDERAL = 128  # federated model input size (if FL model uses 128; update if needed)

# Default labels
class_labels = [
    'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
]

# Try to read metadata and set class order robustly
try:
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ham10000', 'HAM10000_metadata.csv')
    if os.path.exists(metadata_path):
        skin_df = pd.read_csv(metadata_path)
        le = LabelEncoder()
        le.fit(skin_df['dx'])
        # label encoder classes must align with your training pipeline; we use it if available
        class_labels = list(le.classes_)
        print("Using class labels from metadata:", class_labels)
except Exception as e:
    print(f"Could not load class labels dynamically, using default: {e}")

disease_names = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

# -----------------------------
# Prediction helpers
# -----------------------------
def preprocess_image(path, target_size):
    img = load_img(path, target_size=(target_size, target_size))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr

def predict_with_model(model_obj, image_path, target_size):
    arr = preprocess_image(image_path, target_size)
    preds = model_obj.predict(arr)
    preds = np.squeeze(preds)
    idx = int(np.argmax(preds))
    label_key = class_labels[idx] if idx < len(class_labels) else None
    name = disease_names.get(label_key, label_key)
    confidence = float(np.max(preds))
    return {
        'label': label_key,
        'name': name,
        'confidence': f"{confidence:.2%}",
        'raw_probs': preds.tolist()
    }

# Backwards-compatible predict() — uses centralized model
def predict(image_path):
    if centralized_model is None:
        return {'diagnosis': 'Model not loaded', 'confidence': '0%'}
    res = predict_with_model(centralized_model, image_path, SIZE_CENTRAL)
    return {'diagnosis': res['name'], 'confidence': res['confidence']}

# -----------------------------
# Analyze route (uploads + choose model)
# -----------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if file in request
    if 'file' not in request.files:
        return redirect(url_for('detection'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('detection'))

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    filepath = filepath.replace('\\', '/')

    # Determine which model to use from a form field 'model'
    # - If model == "federated" -> use federated model (if loaded)
    # - Else -> centralized
    chosen = request.form.get('model', 'centralized')
    if chosen == 'federated' and federated_model is not None:
        res = predict_with_model(federated_model, filepath, SIZE_FEDERAL)
        diagnosis = res['name']
        confidence = res['confidence']
        model_used = 'federated'
    else:
        # fallback to centralized
        res = predict_with_model(centralized_model, filepath, SIZE_CENTRAL) if centralized_model is not None else None
        if res is None:
            diagnosis = "Model not loaded"
            confidence = "0%"
        else:
            diagnosis = res['name']
            confidence = res['confidence']
        model_used = 'centralized'

    address = request.form.get('user_address', 'Unknown')
    return render_template('index.html',
                           prediction=diagnosis,
                           confidence=confidence,
                           filepath=f'/uploads/{file.filename}',
                           user_address=address,
                           model_used=model_used)

# -----------------------------
# API endpoints for direct programmatic calls
# -----------------------------
@app.route('/predict_centralized', methods=['POST'])
def predict_centralized_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    tmp = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(tmp)
    if centralized_model is None:
        return jsonify({"error": "centralized model not loaded"}), 500
    res = predict_with_model(centralized_model, tmp, SIZE_CENTRAL)
    return jsonify({"model": "centralized", "result": res})

@app.route('/predict_federated', methods=['POST'])
def predict_federated_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    tmp = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(tmp)
    if federated_model is None:
        return jsonify({"error": "federated model not loaded"}), 500
    res = predict_with_model(federated_model, tmp, SIZE_FEDERAL)
    return jsonify({"model": "federated", "result": res})

# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)