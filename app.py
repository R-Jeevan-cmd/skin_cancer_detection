import os
from flask import Flask, request, redirect, render_template, send_from_directory, jsonify, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
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

# Define the SPP layer so that the model can be loaded correctly
def spatial_pyramid_pooling(inputs, levels=[1, 2, 4]):
    shape = inputs.shape
    pool_list = []
    
    for level in levels:
        # Calculate pool size for this level
        pool_size = (int(np.ceil(shape[1] / level)), int(np.ceil(shape[2] / level)))
        
        # Apply max pooling
        x = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_size, padding='same')(inputs)
        
        # Flatten the output
        pooled = tf.keras.layers.Flatten()(x)
        pool_list.append(pooled)
    
    # Concatenate the pooled features
    return tf.keras.layers.Concatenate()(pool_list)

# Load the trained model with custom objects
MODEL_PATH = 'models/skin_lesion_model.h5'
custom_objects = {
    'spatial_pyramid_pooling': spatial_pyramid_pooling
}
model = load_model(MODEL_PATH, custom_objects=custom_objects)

# Ensure the upload folder exists
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

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return redirect(url_for('detection'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('detection'))
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # Convert backslashes to forward slashes
        filepath = filepath.replace('\\', '/')
        prediction_result = predict(filepath)
        
        address = request.form.get('user_address', 'Unknown')
        return render_template('index.html', 
                              prediction=prediction_result['diagnosis'],
                              confidence=prediction_result['confidence'],
                              filepath=f'/uploads/{file.filename}',
                              user_address=address)

SIZE = 64  
# Define the class labels in the same order as they were during training
# These should match the order from the LabelEncoder in trainModel.py
class_labels = [
    'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
]

# Verify that the order matches what was used during training
# This is a more robust approach that will work even if the order changes
try:
    metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ham10000', 'HAM10000_metadata.csv')
    if os.path.exists(metadata_path):
        skin_df = pd.read_csv(metadata_path)
        le = LabelEncoder()
        le.fit(skin_df['dx'])
        class_labels = list(le.classes_)
        print("Using class labels from metadata:", class_labels)
except Exception as e:
    print(f"Could not load class labels dynamically, using default: {e}")
    # Continue with the hardcoded labels

# Add this mapping dictionary after class_labels
disease_names = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

def predict(image_path):
    img = load_img(image_path, target_size=(SIZE, SIZE)) 
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions)]
    full_name = disease_names[predicted_class]
    confidence = float(np.max(predictions))
    return {
        'diagnosis': full_name,
        'confidence': f"{confidence:.2%}"
    }

if __name__ == '__main__':
    app.run(debug=True)