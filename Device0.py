import numpy as np
import os
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# Define the mapping from 4-bit sequences to integers from 1 to 16
base_current_map = {
    '0000': 5.13e-07,
    '0001': 2.47e-06,
    '0010': 8.45e-07,
    '1000': 6.22e-07,
    '1001': 4.81e-06,
    '1010': 5.18e-06,
    '0011': 6.28e-06,
    '0100': 3.45e-06,
    '0101': 6.04e-06,
    '0110': 5.83e-06,
    '0111': 7.82e-06,
    '1011': 8.18e-06,
    '1100': 5.64e-06,
    '1101': 6.64e-06,
    '1110': 6.57e-06,
    '1111': 8.13e-06
}

def read_and_transform_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    currents = [base_current_map[line.strip()]
                for line in lines if line.strip() in base_current_map]
    return currents

def load_data(directory):
    features = []
    labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            currents = read_and_transform_data(file_path)
            label = int(re.search(r'label_(\d+)_', file_name).group(1))
            features.append(currents)
            labels.append(label)
    return np.array(features), np.array(labels)

X, y = load_data(r'C:\Users\97843\Desktop\MNIST\Processed_20')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (80%) and temporary (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Split the temporary set into validation (10%) and test (10%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = Sequential([
    Input(shape=(196,)),
    Dense(10, activation='softmax', kernel_regularizer=l2(0.001))
])

model.compile(optimizer=Adam(learning_rate=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and store the history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
