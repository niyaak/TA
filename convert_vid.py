import cv2
import os

# Path video input
video_path = 'D:/Documents/Github/WahanaSiKecil/segmentasi/vidio/demo9.Mov'
# Folder untuk menyimpan dataset gambar
output_folder = 'D:/Documents/Github/WahanaSiKecil/segmentasi/dataset'
# Interval frame (misal, setiap 10 frame)
frame_interval = 20

# Membuat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Membuka video
cap = cv2.VideoCapture(video_path)

# Cek apakah video berhasil dibuka
if not cap.isOpened():
    print("Error: Video tidak dapat dibuka.")
    exit()

frame_count = 0
# Mulai dari image_62
image_count = 3903

# Membaca frame dari video
while True:
    ret, frame = cap.read()

    # Jika frame tidak berhasil dibaca, keluar dari loop
    if not ret:
        break

    # Simpan setiap interval frame yang diinginkan
    if frame_count % frame_interval == 0:
        # Menyimpan frame sebagai gambar
        image_path = os.path.join(output_folder, f'image_{image_count}.jpg')
        cv2.imwrite(image_path, frame)
        print(f"Frame {frame_count} disimpan sebagai {image_path}")
        image_count += 1

    frame_count += 1

# Menutup video
cap.release()
print("Proses pembuatan dataset gambar selesai.")
