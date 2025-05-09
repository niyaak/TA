import cv2
import math

# Variabel global untuk menyimpan titik
points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", img)

        if len(points) == 2:
            # Hitung jarak Euclidean antar dua titik
            pixel_distance = math.dist(points[0], points[1])
            print(f"Jarak piksel antara dua titik: {pixel_distance:.2f}")
            
            # Input panjang sebenarnya dalam cm (misal: dari 0 ke 10 cm)
            real_cm = float(input("Masukkan panjang sebenarnya antara dua titik (dalam cm): "))
            cm_per_pixel = real_cm / pixel_distance
            print(f"Konversi: {cm_per_pixel:.4f} cm/piksel")

# Load gambar dari file (pastikan penggaris terlihat)
img = cv2.imread("penggaris.jpg")  # Ganti dengan path gambar Anda
if img is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Pastikan path benar.")

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
