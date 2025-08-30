import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import argparse
import os


def create_model(input_shape=(150, 150, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


parser = argparse.ArgumentParser(description='Train Dogs vs Cats CNN')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
args = parser.parse_args()

X = np.load('../data/processed/X.npy')
y = np.load('../data/processed/y.npy')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(X_val, y_val))

os.makedirs('models', exist_ok=True)
model.save('../models/dogs_cats_cnn.keras')
print("Model training complete and saved to models/dogs_cats_cnn.keras")
