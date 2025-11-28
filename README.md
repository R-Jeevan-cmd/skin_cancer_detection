# 🩺 Skin Cancer Detection – Deep Learning Based Web Application

A simple and effective **skin cancer / skin lesion classification web app** built using **Flask + Deep Learning (CNN)**.  
Users can upload a skin-lesion image, and the model predicts the lesion type based on the **HAM10000** dataset.

This project is for **research & educational purposes only** and is **not a medical diagnostic tool**.

---

## 🚀 Features
- Image upload interface using **Flask Web App**
- Deep-learning based classifier using **TensorFlow / Keras**
- Preprocessed HAM10000 dataset
- Modular code structure for training, testing & utilities
- Optional **ResNet50** and **Federated Learning** code

---

## 🧰 Technologies Used

| Area | Technology |
|------|------------|
| Backend | Python, Flask |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV / PIL |
| Dataset | HAM10000 |
| UI | HTML, CSS (Flask Templates) |

---

## 📁 Project Structure

skin_cancer_detection/
├── app.py                  # Flask web application
├── trainModel.py           # CNN model training script
├── resnet50_fl.py          # ResNet50-based model (optional)
├── federated_simulation.py # Federated learning simulation (optional)
├── ham_sort.py             # Dataset sorting/utility script
├── split_data.py           # Data splitting for train/val/test
├── requirements.txt        # All dependencies
├── templates/              # HTML templates
│   └── index.html
├── static/                 # Static assets (CSS/images)
├── data/                   # Dataset folder
│   └── ham10000/           # Place dataset images here
└── README.md               # Documentation

---

## 🔧 Installation & Setup

### 1️⃣ Clone the repository
git clone https://github.com/R-Jeevan-cmd/skin_cancer_detection.git
cd skin_cancer_detection

2️⃣ Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add the dataset
Download the HAM10000 dataset from Kaggle.

Place images inside:
data/ham10000/

▶️ Running the Application

Start Flask server:
python app.py

Open in browser:
👉 http://127.0.0.1:5000/

Upload skin lesion → model predicts the class.

🧪 Training the Model

To retrain the CNN model:
python trainModel.py

What training includes:
	•	Loading dataset
	•	Preprocessing & augmentation
	•	CNN model creation
	•	Training & validation
	•	Saving model weights

If using ResNet50:
python resnet50_fl.py

If testing Federated Learning:
python federated_simulation.py

📈 Model Details

Dataset (HAM10000)
	•	10,000+ dermatoscopic images
	•	Includes 7 lesion classes:
	•	Melanoma
	•	Melanocytic Nevi
	•	Basal Cell Carcinoma
	•	Benign Keratosis
	•	Dermatofibroma
	•	Vascular Lesions
	•	Actinic Keratoses

Preprocessing
	•	Resize images
	•	Normalize pixel values
	•	Split using split_data.py
	•	Augment data (if implemented)

Architecture
	•	Default: Custom CNN
	•	Optional: ResNet50 Transfer Learning

Evaluation (Update with your values)
Metric
Score
Accuracy
—
Loss
—
F1-Score
—

🧑‍💻 How to Use
	1.	Run Flask app
	2.	Upload an image
	3.	Wait for model prediction
	4.	Read classification result
	5.	(Optional) Train model again with more data

⸻

⚠️ Disclaimer

This project is not for medical use.
It is intended only for academic and experimental purposes.

⸻

📝 Future Improvements
	•	Add Grad-CAM heatmap visualization
	•	API endpoints (REST)
	•	Mobile-friendly UI
	•	Improve model accuracy
	•	ONNX/TFLite conversion
	•	Deploy using Docker or Render

⸻

🤝 Contributing
	1.	Fork the repository
	2.	Create a feature branch:
  git checkout -b feature-name
  3.	Commit changes
	4.	Push to GitHub
	5.	Open a Pull Request
  
📚 References
	•	HAM10000 Dataset
	•	TensorFlow Documentation
	•	Flask Documentation
	•	Dermatology Research Papers

⸻

