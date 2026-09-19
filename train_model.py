import tensorflow as tf
import os

# ==========================================
# 1. Load CIFAR-10 Dataset
# ==========================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print("Dataset loaded successfully!")
print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)


# ==========================================
# 2. Preprocess Dataset
# ==========================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\nPreprocessing completed!")
print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())


# ==========================================
# 3. Create CNN Model
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(32, 32, 3)),

    # First Convolution Block
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second Convolution Block
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten feature maps
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(128, activation="relu"),

    # Dropout
    tf.keras.layers.Dropout(0.5),

    # Output layer
    tf.keras.layers.Dense(10, activation="softmax")
])


# ==========================================
# 4. Display Model Architecture
# ==========================================

print("\n==========================================")
print("CNN MODEL ARCHITECTURE")
print("==========================================")

model.summary()


# ==========================================
# 5. Compile Model
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 6. Train Model
# ==========================================

print("\n==========================================")
print("STARTING MODEL TRAINING")
print("==========================================")

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)


# ==========================================
# 7. Evaluate Model
# ==========================================

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")


# ==========================================
# 8. Save Trained Model
# ==========================================

os.makedirs("model", exist_ok=True)

model.save("model/cifar10_cnn.keras")

print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY!")
print("==========================================")
print("Saved file: model/cifar10_cnn.keras")