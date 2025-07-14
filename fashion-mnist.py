# install and import necessary libraries
import cv2
import time
import numpy as np
from keras import datasets, layers, models
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import matplotlib.pyplot as plt

np.random.seed(125)

print("Fashion MNIST Dataset")
time.sleep(2)

# defining the objects in the dataset
names = ["T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boots"]

# load the dataset
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# normalize the dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape the images to ensure they are all grayscale ( 28 x 28 )
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# convert the labels to one-hot encoded
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# model definition
model = models.Sequential()

# first CNN layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2, 2))

# second CNN layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

# third CNN layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# flattening to ensure it can process 1d image
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))

# dropout to prevent overfitting
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation='softmax'))

# compiling the model, metrics and also, enhancing for multi-classification
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# setting up early stopping
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)
# fitting the model
fit = model.fit(train_images, train_labels, batch_size=64, epochs=30,
                validation_data=(test_images, test_labels), callbacks=[early_stop])

# evaluating model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc:.3f}")

# visualization of data
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(fit.history['accuracy'], label='Training Accuracy')
plt.plot(fit.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(fit.history['loss'], label='Training Loss')
plt.plot(fit.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

predictions = model.predict(test_images)

predicted = np.argmax(predictions[0])
true = np.argmax(test_labels[0])
predicted_name = names[predicted]
true_name = names[true]

print(f"Predicted for first test image: {predicted_name}")
print(f"True name for test image: {true_name}")


def live_camera():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Error! The webcam isn't open yet.")
        return

    # background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

    # initialize CLAHE for contrast adjustment
    contrast_adj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to capture the frame.")
            break
        # resizing for consistent framing
        display_frame = cv2.resize(frame, (640, 480))

        # defining the bounding box
        box_size = 200
        center_x, center_y = 640 // 2, 480 // 2
        x1 = max(center_x - box_size // 2, 0)
        y1 = max(center_y - box_size // 2, 0)
        x2 = min(center_x + box_size // 2, 640)
        y2 = min(center_y + box_size // 2, 480)

        roi = display_frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        fg_mask = bg_subtractor.apply(gray)

        enhanced = contrast_adj.apply(gray)

        processed = cv2.bitwise_and(enhanced, enhanced, mask=fg_mask)

        resized_image = cv2.resize(processed, (28, 28))
        normalized_image = resized_image / 255.0
        input_frame = np.expand_dims(normalized_image, axis=-1)
        input_frame = np.expand_dims(input_frame, axis=0)

        predictions = model.predict(input_frame, verbose=0)
        predicted = np.argmax(predictions[0])
        confidence_score = np.max(predictions[0]) * 100
        predicted_name = names[predicted]

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(display_frame, f"Predicted: {predicted_name}, ({confidence_score:.2f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Clothing Item Webcam Prediction', display_frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()


live_camera()
