import cv2
import math

# Inisialisasi variabel
points = []
calibrated = False

def click_event(event, x, y, flags, param):
    global points, calibrated
    if event == cv2.EVENT_LBUTTONDOWN and not calibrated:
        points.append((x, y))
        if len(points) == 2:
            # Hitung jarak piksel antara dua titik
            pixel_distance = math.dist(points[0], points[1])
            print(f"Jarak piksel: {pixel_distance:.2f}")
            real_cm = float(input("Masukkan panjang sebenarnya antara dua titik (cm): "))
            cm_per_pixel = real_cm / pixel_distance
            print(f"Konversi: {cm_per_pixel:.4f} cm/piksel")
            calibrated = True  # Set agar tidak terus-menerus kalibrasi ulang

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Kamera tidak dapat dibuka.")

cv2.namedWindow("Kalibrasi")
cv2.setMouseCallback("Kalibrasi", click_event)

print("Klik dua titik pada penggaris di tampilan video untuk kalibrasi...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tampilkan titik jika sudah diklik
    for pt in points:
        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

    cv2.imshow("Kalibrasi", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Tekan ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
