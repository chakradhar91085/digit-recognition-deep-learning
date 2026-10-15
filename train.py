"""Train a small digit classifier on MNIST and save it to models/digits.keras."""
from tensorflow import keras
from tensorflow.keras import layers

(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
x_tr, x_te = x_tr[..., None] / 255.0, x_te[..., None] / 255.0

model = keras.Sequential([
    layers.Input((28, 28, 1)),
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomZoom(0.1),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPool2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPool2D(),
    layers.Flatten(), layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_tr, y_tr, epochs=8, batch_size=128, validation_split=0.1, verbose=2)
print("test accuracy: %.4f" % model.evaluate(x_te, y_te, verbose=0)[1])
model.save("models/digits.keras")
