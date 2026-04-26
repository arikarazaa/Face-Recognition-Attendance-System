import os
import tensorflow as tf


def convert(h5_path='facial_model.h5', output_path='facial_model.tflite'):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Trained model not found at '{h5_path}'. Run train.py first."
        )

    print(f"Loading model from '{h5_path}'...")
    model = tf.keras.models.load_model(h5_path)

    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ Saved '{output_path}'  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    convert()