import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# ==== Konfigurasi ====


# Path dasar ke folder image & label
base_image_path = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/3/test'

# Nama file (cukup diganti saja bagian ini)
file_name = 'image_722_jpg.rf.0b7adfb2a5a8a636d996798a1bef91d5'

# Path lengkap image dan label (label diasumsikan PNG)
test_image_path = os.path.join(base_image_path, file_name + '.jpg')
ground_truth_path = os.path.join(base_image_path, file_name + '_mask.png')


best_model_path = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/cobamodelbagusbismillah.keras'

# Ukuran input model
img_height, img_width = 256, 448
num_classes = 3  # Ubah sesuai jumlah kelas Anda

# ==== Fungsi Preprocessing ====
def preprocess_image(image_path, target_size):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0), image_array  # (1, H, W, C), (H, W, C)

def preprocess_mask(mask_path, target_size):
    mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=target_size, color_mode='grayscale')
    mask_array = tf.keras.preprocessing.image.img_to_array(mask).astype(np.uint8)
    mask_array = np.squeeze(mask_array, axis=-1)  # (H, W)
    return mask_array

def calculate_pixel_accuracy(gt_mask, pred_mask):
    correct = np.sum(gt_mask == pred_mask)
    total = gt_mask.size
    return correct / total

def visualize_result(original_image, predicted_mask, ground_truth_mask, pixel_accuracy):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image)

    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(predicted_mask, cmap='jet')

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Mask")
    plt.imshow(ground_truth_mask, cmap='jet')

    plt.suptitle(f"Pixel-wise Accuracy: {pixel_accuracy:.4f}", fontsize=14)
    plt.show()

# ==== Validasi Path ====
for path in [test_image_path, ground_truth_path, best_model_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found: {path}")

# ==== Load Model dan Proses ====
print("Loading the best model...")
model = tf.keras.models.load_model(best_model_path)

print("Preprocessing test image and mask...")
input_image, original_image = preprocess_image(test_image_path, (img_height, img_width))
ground_truth_mask = preprocess_mask(ground_truth_path, (img_height, img_width))

print("Running prediction...")
prediction = model.predict(input_image)
predicted_mask = np.argmax(prediction[0], axis=-1)  # Shape: (H, W)

pixel_accuracy = calculate_pixel_accuracy(ground_truth_mask, predicted_mask)
print(f"Pixel-wise Accuracy: {pixel_accuracy:.4f}")

print("Visualizing result...")
visualize_result(original_image, predicted_mask, ground_truth_mask, pixel_accuracy)
