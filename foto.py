import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Load the trained model
model = load_model('D:/Documents/Github/TA/segmentation_model.h5', 
                   custom_objects={'iou_metric': iou_metric})

# Load and preprocess the image
image_path = 'D:/Documents/Github/TA/dataset/image_12.jpg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to the input dimensions of the model
proc_img = cv2.resize(img, (848, 480))  
proc_img = np.expand_dims(proc_img, axis=0) / 255.0  # Add batch dimension and normalize

# Model prediction
pred = model.predict(proc_img)
pred_mask = np.argmax(pred[0], axis=-1)  # Predicted class per pixel

# Define colors for each class
colors = {
    0: [0, 0, 0],          # background
    1: [250, 250, 55],     # indopoint
    2: [51, 221, 255],     # palang
    3: [102, 255, 102],    # road
    4: [255, 0, 204]       # trotoar
}

# Create a color-coded output image based on the predicted mask
output_img = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
for class_idx, color in colors.items():
    output_img[pred_mask == class_idx] = color  # Apply color to each class

# Display the original image and the color-coded output
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(img)
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(output_img)
axs[1].set_title("Predicted Segmentation")
axs[1].axis('off')

# Show and save the final result
plt.show()
fig.savefig('D:/Documents/Github/TA/output/output.png', bbox_inches="tight", dpi=300)
