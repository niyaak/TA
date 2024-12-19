import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import os
import numpy as np

# Define paths and parameters
image_dir = 'D:/Documents/Github/TA/datasetsegmentasi'
mask_dir = 'D:/Documents/Github/TA/label/SegmentationClass'
img_height, img_width = 480, 848
batch_size = 4  # Reducing batch size to manage memory
num_classes = 5  # Number of segmentation classes

# Function to load and preprocess data
def load_data(image_dir, mask_dir, img_height, img_width):
    images = []
    masks = []

    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace('.jpg', '.png'))

        # Load and normalize the image
        img = load_img(img_path, target_size=(img_height, img_width))
        img = img_to_array(img) / 255.0
        images.append(img)

        # Load the mask and map colors to class indices
        mask = load_img(mask_path, target_size=(img_height, img_width))
        mask = img_to_array(mask).astype(np.uint8)  # Convert to integer type

        # Create an empty array for the label indices
        label_mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Map each color in the mask to a corresponding class index
        label_mask[(mask == [0, 0, 0]).all(axis=-1)] = 0           # background
        label_mask[(mask == [250, 250, 55]).all(axis=-1)] = 1      # indopoint
        label_mask[(mask == [51, 221, 255]).all(axis=-1)] = 2      # palang
        label_mask[(mask == [102, 255, 102]).all(axis=-1)] = 3     # road
        label_mask[(mask == [255, 0, 204]).all(axis=-1)] = 4       # trotoar

        masks.append(label_mask)

    images = np.array(images)
    masks = np.array(masks)
    return images, masks

# U-Net Model Definition
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

# Custom IoU Metric
def iou_metric(y_true, y_pred):
    # Convert y_pred to class indices
    y_pred = tf.argmax(y_pred, axis=-1)  # Convert to class indices
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # Reshape y_true to match y_pred's shape
    y_true = tf.squeeze(y_true, axis=-1)

    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return intersection / (union + 1e-10)  # Avoid division by zero

# Data loading and preprocessing
images, masks = load_data(image_dir, mask_dir, img_height, img_width)

# Expand mask dimensions to match model output (batch_size, height, width, 1)
masks = np.expand_dims(masks, axis=-1)

# Create the model
model = unet_model()

# Compile the model with custom IoU metric
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', iou_metric])

# Create data augmentation generators
data_gen_args = dict(rotation_range=20,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='nearest')
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Flow data through generators
image_generator = image_datagen.flow(images, batch_size=batch_size, seed=42)
mask_generator = mask_datagen.flow(masks, batch_size=batch_size, seed=42)

# Combine generators
train_generator = zip(image_generator, mask_generator)

# Train the model
model.fit(train_generator, steps_per_epoch=len(images) // batch_size, epochs=50)

# Save the model
model.save('unet_segmentation_model.h5')
