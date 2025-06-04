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
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array, load_img

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
IMAGE_SIZE = (224, 224)  # Resize to fit VGG16 input size

print("Emotion Categories:", EMOTIONS)

# Function to generate images and labels
def image_generator(input_path, emotions, image_size):
    for index, emotion in enumerate(emotions):
        for filename in os.listdir(os.path.join(input_path, emotion)):
            img = cv2.imread(os.path.join(input_path, emotion, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)  # Resize image for VGG16
            img = img_to_array(img)  # Convert to array
            img = img / 255.0  # Normalize image
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

# Load the pre-trained VGG16 model (excluding top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model to avoid updating them during training
for layer in base_model.layers:
    layer.trainable = False

# Create the model by adding custom layers on top of the base model
model = Sequential()
model.add(base_model)  # Add the base model (VGG16)
model.add(Flatten())  # Flatten the output of the base model
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))  # Add a dense layer
model.add(Dropout(0.5))  # Add dropout
model.add(Dense(6, activation='softmax'))  # Output layer for 6 classes

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model for 20 epochs
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=128,
                    callbacks=[EarlyStopping(patience=5, monitor='val_loss'),
                               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)])

# Select 6 random test samples
num_samples = 6
random_indices = np.random.choice(X_test.shape[0], num_samples, replace=False)

# Make predictions on the selected test samples
y_pred = model.predict(X_test[random_indices])
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
