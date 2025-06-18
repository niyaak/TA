import tensorflow as tf
import tf2onnx

# Path ke model Keras asli Anda
KERAS_MODEL_PATH = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/datasetbener3coba2.keras'
# Path untuk menyimpan model ONNX baru
ONNX_MODEL_PATH = 'model2_onnx.onnx'

print("Memuat model Keras...")
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

# =================================================================
# ====== INI BAGIAN PENTING YANG DITAMBAHKAN / DIPERBAIKI ======
# =================================================================

# 1. Tentukan spesifikasi input model Anda secara manual.
#    Bentuknya adalah [batch_size, height, width, channels]
#    Gunakan 'None' untuk batch_size agar fleksibel.
#    Pastikan height, width, dan channels sesuai dengan model Anda.
input_signature = [
    tf.TensorSpec(shape=[None, 224, 384, 3], dtype=tf.float32, name="input_layer")
]


print(f"Mengonversi model ke ONNX dengan input signature eksplisit...")

# 2. Tambahkan argumen 'input_signature' saat memanggil fungsi konversi
model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=input_signature, 
    opset=13
)
# =================================================================


print(f"Menyimpan model ONNX ke {ONNX_MODEL_PATH}")
with open(ONNX_MODEL_PATH, "wb") as f:
    f.write(model_proto.SerializeToString())

print("="*50)
print("Konversi ke ONNX selesai dengan sukses.")
print("="*50)