import tensorflow as tf

model_path = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/datasetbener3coba.keras'
model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Kuantisasi float16
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Simpan model
with open("model_quant.tflite", "wb") as f:
    f.write(tflite_model)
