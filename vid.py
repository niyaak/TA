import cv2
import numpy as np
import tensorflow as tf

# === Konfigurasi ===
model_path = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/datasetbener3coba.keras'  # Path ke model
input_video_path = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/vidio/tes/democpt2.mp4'
output_video_path = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/vidio/tes/output4.mp4'

img_height, img_width = 256, 448   # Ukuran input model

# Daftar warna sesuai jumlah kelas model (0–4)
class_colors = {
    0: (0, 0, 0),         # background - hitam
    1: (0, 255, 0),       # jalan - hijau
    2: (255, 0, 0),       # tepikanan - merah
}

# === Load model ===
print("Memuat model...")
model = tf.keras.models.load_model(model_path)

# === Buka video ===
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Gagal membuka video input.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print("Memproses video...")
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize ke ukuran input model
    resized_frame = cv2.resize(frame, (img_width, img_height))
    normalized = resized_frame.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)  # Shape: (1, H, W, 3)

    # Prediksi mask
    pred = model.predict(input_tensor)[0]  # Shape: (H, W, num_classes)
    pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)  # Shape: (H, W)

    # Resize pred_mask ke ukuran asli frame
    mask_resized = cv2.resize(pred_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    # Buat overlay warna
    overlay = np.zeros_like(frame)
    for class_id, color in class_colors.items():
        overlay[mask_resized == class_id] = color

    # Gabungkan dengan frame asli
    combined = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Tulis frame hasil ke video output
    out.write(combined)

    # Debug log tiap beberapa frame
    frame_count += 1
    if frame_count % 30 == 0:
        unique_classes = np.unique(mask_resized)
        print(f"Frame ke-{frame_count}: kelas terdeteksi = {unique_classes}")

# === Bersihkan ===
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video hasil disimpan di: {output_video_path}")
