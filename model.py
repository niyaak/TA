import cv2
import numpy as np
import time
import tensorflow as tf

# === Konfigurasi Model ===
model_path = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/cobamodelbagusbismillah.keras'
model = tf.keras.models.load_model(model_path)

img_height, img_width = 256, 448   # Sesuaikan dengan input model Anda

# Class colors (update sesuai urutan labelmu: background, jalan, tepikanan, tepikiri, percabangan)
class_colors = {
    0: (0, 0, 0),           # background - hitam
    1: (102, 255, 102),     # jalan - hijau muda
    2: (255, 0, 0),         # tepikanan - merah
}

# Ukuran input model
input_height = 256
input_width = 448

# Mulai webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize untuk model
    resized_frame = cv2.resize(frame, (input_width, input_height))
    input_image = resized_frame.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_image, axis=0)

    # Timing inference
    start_time = time.time()
    pred = model.predict(input_tensor, verbose=0)[0]
    inference_time = (time.time() - start_time) * 1000  # in ms

    # Argmax untuk ambil class index
    pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)

    # Konversi ke RGB
    seg_rgb = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    for class_idx, color in class_colors.items():
        seg_rgb[pred_mask == class_idx] = color

    # Resize segmen ke ukuran asli frame
    seg_rgb = cv2.resize(seg_rgb, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay
    overlay = cv2.addWeighted(frame, 0.6, seg_rgb, 0.4, 0)

    # Tampilkan waktu inference
    fps = 1000 / inference_time if inference_time > 0 else 0
    cv2.putText(overlay, f"Inference: {inference_time:.1f} ms | FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Tampilkan hasil
    cv2.imshow("Segmentasi Real-Time", overlay)

    # Tombol keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()