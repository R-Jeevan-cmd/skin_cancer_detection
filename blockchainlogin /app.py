from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from web3 import Web3
from eth_account.messages import encode_defunct

app = Flask(__name__, template_folder="templates")
CORS(app)

# Connect to Ethereum Node (Ganache or Infura)
INFURA_URL = "http://127.0.0.1:7545"  # Use Infura URL for Testnet if needed
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

def verify_signature(address, message, signature):
    """Verifies an Ethereum signed message."""
    message_encoded = encode_defunct(text=message)
    recovered_address = web3.eth.account.recover_message(message_encoded, signature=signature)
    
    return recovered_address.lower() == address.lower()

@app.route('/')
def home():
    return render_template("index.html")  # Serve the login page

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

    return jsonify({"redirect": url_for('welcome', address=user_address)})

@app.route('/welcome')
def welcome():
    address = request.args.get("address", "Unknown")
    return render_template('welcome.html', address=address)

if __name__ == '__main__':
    app.run(debug=True)
