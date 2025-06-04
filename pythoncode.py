import random
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense, MaxPool2D
from keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

# Define the path to the organized dataset with 6 classes
base_path = r'F:\3rd semester\Image Processing\project\code\affectnet_6class'

# Define the list of 6 categories
categories = ['happy', 'sad', 'surprise', 'angry', 'fear', 'neutral']

# Set up the plot for displaying images
fig, axs = plt.subplots(2, 3, sharey=True, constrained_layout=True, 
                        figsize=(8, 8), dpi=80, facecolor='gray', edgecolor='k')
fig.suptitle("Sample Faces and Labels")
axs = axs.flatten()

# Display one random image from each of the 6 categories
for i, category in enumerate(categories):
    category_path = os.path.join(base_path, category)
    if os.path.exists(category_path):
        images = os.listdir(category_path)
        img_name = random.choice(images)
        img_path = os.path.join(category_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].imshow(img)
        axs[i].set_title(category)
        axs[i].axis('off')
    else:
        axs[i].text(0.5, 0.5, f"Category {category} not found", fontsize=12, ha='center', va='center')
        axs[i].axis('off')

plt.show()

# List of emotion categories from the dataset
INPUT_PATH = r'F:\3rd semester\Image Processing\project\code\affectnet_6class'
EMOTIONS = [f.name for f in os.scandir(INPUT_PATH) if f.is_dir()]
IMAGE_SIZE = (96, 96)

print("Emotion Categories:", EMOTIONS)

# Function to generate images and labels
def image_generator(input_path, emotions, image_size):
    for index, emotion in enumerate(emotions):
        for filename in os.listdir(os.path.join(input_path, emotion)):
            img = cv2.imread(os.path.join(input_path, emotion, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img, index

# Function to load images and labels
def load_images(input_path, emotions, image_size):
    X, y = [], []
    for img, label in image_generator(input_path, emotions, image_size):
        X.append(img)
        y.append(label)
    X = np.array(X)
    y = to_categorical(np.array(y))
    return X, y

# Load the images
X, y = load_images(INPUT_PATH, EMOTIONS, IMAGE_SIZE)
input_shape = X[0].shape

# Train-test split of the pre-processed data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Create the CNN model
model_4 = Sequential()

model_4.add(Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.3))

model_4.add(Conv2D(64, (3,3), activation="relu"))
model_4.add(BatchNormalization())
model_4.add(Conv2D(64, (3,3), activation="relu"))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.4))

model_4.add(Conv2D(128, (3,3), activation="relu"))
model_4.add(BatchNormalization())
model_4.add(Conv2D(128, (3,3), activation="relu"))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.5))

model_4.add(Conv2D(256, (3,3), activation="relu"))
model_4.add(BatchNormalization())
model_4.add(Conv2D(256, (3,3), activation="relu"))
model_4.add(BatchNormalization())
model_4.add(MaxPool2D(pool_size=(2,2)))
model_4.add(Dropout(0.6))

model_4.add(Flatten())
model_4.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model_4.add(BatchNormalization())
model_4.add(Dropout(0.5))
model_4.add(Dense(6, activation='softmax'))

# Compile the model
model_4.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model_4.summary()

# Train the model
history = model_4.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=128,
                      callbacks=[EarlyStopping(patience=5, monitor='val_loss'),
                                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)])

# Select 6 random test samples
num_samples = 6
random_indices = np.random.choice(X_test.shape[0], num_samples, replace=False)

# Make predictions on the selected test samples
y_pred = model_4.predict(X_test[random_indices])
predicted_classes = np.argmax(y_pred, axis=1)
true_classes = np.argmax(y_test[random_indices], axis=1)

# Class labels
class_labels = ['surprise', 'fear', 'angry', 'neutral', 'sad', 'happy']

# Plot the results
fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(X_test[random_indices[i]].astype('uint8'))  # Display the image
    ax.set_title(f"Predicted: {class_labels[predicted_classes[i]]}\nActual: {class_labels[true_classes[i]]}")
    ax.axis('off')  # Hide the axis

plt.tight_layout()
plt.show()

