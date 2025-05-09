import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pandas as pd
import numpy as np

# Memastikan hanya CPU yang digunakan, nonaktifkan GPU
tf.config.set_visible_devices([], 'GPU')

# Define paths and parameters
base_dir = 'D:\Documents\Github\WahanaSiKecil\segmentasi\datasett'  # Update dengan path dataset Anda
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')

# Parameter utama
img_height, img_width = 432, 768  # Resolusi gambar yang lebih kecil untuk menghemat RAM
batch_size = 4

# Function to load class mapping from _classes.csv
def load_classes(folder):
    classes_file = os.path.join(folder, '_classes.csv')
    print(f"Loading classes from {classes_file}...")
    df = pd.read_csv(classes_file)

    # Handle column names dynamically
    pixel_col = [col for col in df.columns if 'Pixel' in col][0]
    class_col = [col for col in df.columns if 'Class' in col][0]

    # Ensure proper mapping
    class_mapping = {}
    for idx, row in df.iterrows():
        pixel_value = int(row[pixel_col])
        class_name = row[class_col].strip()
        class_mapping[pixel_value] = idx

    print(f"Loaded class mapping: {class_mapping}")
    return class_mapping

# Function to load and preprocess data
def load_data(folder):
    class_mapping = load_classes(folder)  # Load class mapping for the folder
    images = []
    masks = []

    for file_name in os.listdir(folder):
        if file_name.endswith('.jpg'):
            img_path = os.path.join(folder, file_name)
            mask_path = os.path.join(folder, file_name.replace('.jpg', '_mask.png'))

            # Load and normalize the image
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img)

            # Load the mask
            mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(img_height, img_width), color_mode='grayscale')
            mask = tf.keras.preprocessing.image.img_to_array(mask).astype(np.uint8).squeeze()

            # Normalize mask pixel values to class indices
            for pixel_value, class_idx in class_mapping.items():
                mask[mask == pixel_value] = class_idx
            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)
    masks = np.expand_dims(masks, axis=-1)  # Expand dimensions for compatibility
    return images, masks, len(class_mapping)

# Load training and validation data
print("Loading training data...")
train_images, train_masks, num_classes_train = load_data(train_dir)
print("Loading validation data...")
val_images, val_masks, num_classes_val = load_data(val_dir)

# Ensure the same number of classes across train and val
if num_classes_train != num_classes_val:
    raise ValueError("The number of classes in training and validation datasets are not the same.")

num_classes = num_classes_train  # Set the number of classes

# Define U-Net model
def unet_model(input_size=(img_height, img_width, 3), num_classes=num_classes):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs, outputs)
    return model

# Build and compile the model
print("Building the model...")
model = unet_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
model.fit(train_images, train_masks,
          validation_data=(val_images, val_masks),
          epochs=50,
          batch_size=batch_size,
          callbacks=[checkpoint])
