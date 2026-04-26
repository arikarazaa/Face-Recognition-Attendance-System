"""
Real-time face recognition attendance system.

Prerequisites:
  1. Run data_prep.py   → generates .npy files + label_encoder.pkl
  2. Run train.py       → generates facial_model.h5
  3. Run convert_tflite.py → generates facial_model.tflite
  4. Place serviceAccountKey.json in the project root
"""

import os
import time
import pickle

import cv2
import numpy as np
import tensorflow as tf

from firebase_rtdb import firebase_init, log_attendance

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT              = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(ROOT, 'facial_model.tflite')
ENCODER_PATH      = os.path.join(ROOT, 'label_encoder.pkl')
SERVICE_ACCOUNT   = os.path.join(ROOT, 'serviceAccountKey.json')

DB_URL          = 'https://ml-mid-default-rtdb.asia-southeast1.firebasedatabase.app'
IMG_SIZE        = (100, 100)
CONF_THRESHOLD  = 0.80          # Minimum confidence to accept a prediction
TZ              = 'Asia/Karachi'
LOG_COOLDOWN    = 10            # Seconds between repeated logs for the same person

# ── Load TFLite model ─────────────────────────────────────────────────────────
if not os.path.exists(TFLITE_MODEL_PATH):
    raise FileNotFoundError(
        f"TFLite model not found at '{TFLITE_MODEL_PATH}'. "
        "Run convert_tflite.py first."
    )

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ── Load label encoder ────────────────────────────────────────────────────────
if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        f"Label encoder not found at '{ENCODER_PATH}'. "
        "Run data_prep.py first."
    )

with open(ENCODER_PATH, 'rb') as f:
    encoder = pickle.load(f)

# ── Firebase init (optional — attendance still shown on screen if it fails) ───
firebase_ready = False
try:
    firebase_init(SERVICE_ACCOUNT, db_url=DB_URL)
    firebase_ready = True
    print("✅ Firebase initialized.")
except Exception as e:
    print(f"⚠️  Firebase unavailable: {e}")
    print("    Attendance will be displayed on screen only.")

# ── Haar cascade face detector ────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR face crop to a normalized float32 tensor
    ready for the TFLite interpreter.

    BUG FIX: original code converted to RGB but then accidentally
    used face_bgr (not face_rgb) for the resize/normalize step,
    causing a silent color-channel mismatch during inference.
    """
    face_rgb     = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)   # BGR → RGB
    face_resized = cv2.resize(face_rgb, IMG_SIZE)               # use face_rgb ✓
    face_norm    = face_resized.astype('float32') / 255.0
    return np.expand_dims(face_norm, axis=0)                    # (1, H, W, 3)


def predict_name(face_bgr: np.ndarray):
    """
    Run TFLite inference on a face crop.

    Returns:
        (name, confidence) — name is "Unknown" if below CONF_THRESHOLD.
    """
    tensor = preprocess_face(face_bgr)
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()

    preds       = interpreter.get_tensor(output_details[0]['index'])[0]
    conf        = float(np.max(preds))
    label_index = int(np.argmax(preds))

    if conf < CONF_THRESHOLD:
        return "Unknown", conf

    name = encoder.inverse_transform([label_index])[0]
    return name, conf


# ── Main recognition loop ─────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera index.")

    print("\nWebcam started. Press 'q' to quit.\n")
    last_log_time: dict[str, float] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face_crop    = frame[y:y + h, x:x + w]
            name, conf   = predict_name(face_crop)
            is_known     = name != "Unknown"
            color        = (0, 200, 0) if is_known else (0, 0, 220)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{name}  {conf:.0%}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )

            if is_known and firebase_ready:
                now = time.time()
                if name not in last_log_time or (now - last_log_time[name]) > LOG_COOLDOWN:
                    log_attendance(name, confidence=conf, tz=TZ)
                    last_log_time[name] = now

        cv2.imshow('Face Recognition Attendance  |  q = quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()