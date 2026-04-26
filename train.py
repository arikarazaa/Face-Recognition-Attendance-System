import numpy as np
import pickle
from model import create_model


def main():
    print("Loading pre-processed data...")
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test  = np.load('X_test.npy')
    y_test  = np.load('y_test.npy')

    print("Loading label encoder...")
    with open('label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    num_classes = len(encoder.classes_)
    print(f"Classes ({num_classes}): {encoder.classes_}")

    print("Creating model...")
    model = create_model(input_shape=(100, 100, 3), num_classes=num_classes)
    model.summary()

    print("\nTraining...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=8,   # Increased from 4 for slightly more stable gradients
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy : {test_acc * 100:.2f}%")
    print(f"Test Loss     : {test_loss:.4f}")

    model.save('facial_model.h5')
    print("Model saved as 'facial_model.h5'")


if __name__ == "__main__":
    main()