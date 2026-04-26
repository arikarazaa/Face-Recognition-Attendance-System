# Face-Recognition-Attendance-System

An automated attendance system that uses a Convolutional Neural Network (CNN) to recognize faces in real-time via webcam and logs attendance to Firebase Realtime Database — no manual roll calls needed.

 Features

 Real-time face detection using OpenCV Haar Cascades
 CNN-based face recognition with TFLite for fast inference
 Automatic attendance logging to Firebase Realtime Database
 Per-person cooldown timer to prevent duplicate entries
 Graceful offline mode if Firebase is unavailable


📁 Project Structure
face-attendance/
├── dataset/                  ← Your training images (you populate this)
│   ├──
│
├── data_prep.py              ← Load & preprocess images → .npy files
├── model.py                  ← CNN architecture definition
├── train.py                  ← Train the model → facial_model.h5
├── convert_tflite.py         ← Convert .h5 → facial_model.tflite
├── export_labels.py          ← Export class names → labels.json
├── firebase_rtdb.py          ← Firebase Realtime Database helpers
├── infer.py                  ← Real-time webcam inference (main app)
├── serviceAccountKey.json    ← Your Firebase credentials (not committed)
└── requirements.txt

⚙️ Setup
1. Clone the repository
bashgit clone https://github.com/your-username/face-attendance.git
cd face-attendance
2. Install dependencies
bashpip install -r requirements.txt
3. Prepare your dataset
Create a dataset/ folder. Inside it, make one subfolder per person named after them, and put their face photos inside:
dataset/
├── Alice/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── Bob/
    ├── photo1.jpg
    └── photo2.jpg

Tips for better accuracy:

Use 20–50 images per person
Vary lighting, angles, and expressions
Make sure the face is clearly visible and well-lit
Avoid blurry or very small images


4. Set up Firebase

Go to Firebase Console and create a project
Enable Realtime Database and set rules to allow authenticated writes
Go to Project Settings → Service Accounts → Generate new private key
Save the downloaded file as serviceAccountKey.json in the project root
Copy your database URL (e.g. https://your-project-id.firebaseio.com) and paste it into infer.py:

pythonDB_URL = 'https://your-project-id.firebaseio.com'

 Never commit serviceAccountKey.json to GitHub. Add it to .gitignore.


 Usage
Run the following scripts in order:
Step 1 — Preprocess the dataset
bashpython data_prep.py
Outputs: X_train.npy, X_test.npy, y_train.npy, y_test.npy, label_encoder.pkl
Step 2 — Train the model
bashpython train.py
Outputs: facial_model.h5
Training runs for 20 epochs. Test accuracy is printed at the end.
Step 3 — Convert to TFLite
bashpython convert_tflite.py
Outputs: facial_model.tflite (optimized for faster inference)
Step 4 — Export labels (optional)
bashpython export_labels.py
Outputs: labels.json — useful for external tools or dashboards.
Step 5 — Run the attendance system
bashpython infer.py
Your webcam opens. Recognized faces are labeled with a name and confidence score. Attendance is logged to Firebase automatically. Press q to quit.

 Firebase Data Structure
Attendance is stored under the following path:
attendance/
└── 2025-07-15/
    └── Alice/
        └── -NxAbCdEfG/
            ├── confidence: 0.94
            └── timestamp:  "2025-07-15T09:03:22+05:00"
Each person gets a new push entry per attendance event (controlled by LOG_COOLDOWN in infer.py).

 Configuration
All key settings are at the top of infer.py:
VariableDefaultDescriptionCONF_THRESHOLD0.80Minimum confidence (0–1) to accept a recognitionLOG_COOLDOWN10Seconds before the same person can be logged againIMG_SIZE(100, 100)Must match the size used during trainingTZ'Asia/Karachi'Timezone for timestamps

 Model Architecture
A lightweight CNN trained from scratch on your dataset:
Input (100×100×3)
    → Conv2D(32, 3×3, ReLU)  → MaxPool
    → Conv2D(64, 3×3, ReLU)  → MaxPool
    → Flatten
    → Dense(128, ReLU)
    → Dropout(0.3)
    → Dense(num_classes, Softmax)

Optimizer: Adam
Loss: Sparse Categorical Crossentropy
Inference: TFLite (converted from Keras .h5)


 .gitignore
Create a .gitignore with at least:
gitignore# Firebase credentials
serviceAccountKey.json

# Trained model files
*.h5
*.tflite
*.pkl
*.npy

# Dataset (usually too large for GitHub)
dataset/

# Python cache
__pycache__/
*.pyc
.env

 Requirements
tensorflow
opencv-python
scikit-learn
firebase-admin
pytz
numpy
joblib
Install all at once:
bashpip install -r requirements.txt

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

📄 License
MIT
