import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(data_dir='dataset', img_size=(100, 100)):
    images, labels = [], []

    print(f"Loading images from '{data_dir}'...")
    for person_name in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        count = 0
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(person_name)
                count += 1
        print(f"  - Found {count} images for '{person_name}'")

    if not images:
        raise ValueError(f"No images found in '{data_dir}'. Check your dataset folder.")

    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels)

    encoder = LabelEncoder()
    labels_enc = encoder.fit_transform(labels)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    print("\nLabel encoder saved to 'label_encoder.pkl'")

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels_enc, test_size=0.2, random_state=42, stratify=labels_enc
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    print(f"\nData processing complete.")
    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Testing  samples : {X_test.shape[0]}")
    print(f"  Image shape      : {X_train.shape[1:]}")
    print("Processed data saved as .npy files.")