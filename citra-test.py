import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Membangun model CNN sederhana
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Kompilasi model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Memuat dataset contoh (MNIST)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalisasi data dan ubah ukuran citra
X_train = X_train / 255.0
X_test = X_test / 255.0

# Melatih model
model.fit(X_train, y_train, epochs=5)

# Evaluasi model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Menggunakan model untuk prediksi pada citra contoh
sample_image = X_test[0]  # Mengambil citra contoh dari dataset
sample_image = np.expand_dims(sample_image, axis=0)  # Menambahkan dimensi batch
predictions = model.predict(sample_image)
predicted_label = np.argmax(predictions)

print(f"Predicted label: {predicted_label}")
