import cv2
import numpy as np
import tensorflow as tf
import skfuzzy as fuzz
import skfuzzy.control as ctrl

# ====== Fuzzy Variable Definitions ======
jt = ctrl.Antecedent(np.arange(0, 121, 1), 'jt')
djt = ctrl.Antecedent(np.arange(-20, 21, 1), 'djt')
theta = ctrl.Consequent(np.arange(-60, 61, 1), 'theta')

# Membership Functions for JT
jt['td'] = fuzz.trapmf(jt.universe, [0, 0, 10, 25])
jt['ad'] = fuzz.trimf(jt.universe, [15, 30, 45])
jt['n']  = fuzz.trimf(jt.universe, [35, 55, 75])
jt['aj'] = fuzz.trimf(jt.universe, [60, 80, 100])
jt['tj'] = fuzz.trapmf(jt.universe, [90, 110, 120, 120])

# Membership Functions for ΔJT
djt['dc'] = fuzz.trapmf(djt.universe, [-20, -20, -12, -8])
djt['dl'] = fuzz.trimf(djt.universe, [-10, -6, -2])
djt['s']  = fuzz.trimf(djt.universe, [-3, 0, 3])
djt['jl'] = fuzz.trimf(djt.universe, [2, 6, 10])
djt['jc'] = fuzz.trapmf(djt.universe, [8, 12, 20, 20])

# Membership Functions for θ
theta['krt'] = fuzz.trapmf(theta.universe, [-60, -60, -50, -40])
theta['krs'] = fuzz.trimf(theta.universe, [-45, -30, -15])
theta['krr'] = fuzz.trimf(theta.universe, [-20, -10, -2])
theta['l']   = fuzz.trimf(theta.universe, [-5, 0, 5])
theta['knr'] = fuzz.trimf(theta.universe, [2, 10, 20])
theta['kns'] = fuzz.trimf(theta.universe, [15, 30, 45])
theta['knt'] = fuzz.trapmf(theta.universe, [40, 50, 60, 60])

# Fuzzy Rules
rules = [
    ctrl.Rule(jt['td'] & djt['dc'], theta['knt']),
    ctrl.Rule(jt['td'] & djt['dl'], theta['kns']),
    ctrl.Rule(jt['td'] & djt['s'],  theta['knr']),
    ctrl.Rule(jt['td'] & djt['jl'], theta['l']),
    ctrl.Rule(jt['td'] & djt['jc'], theta['krr']),
    ctrl.Rule(jt['ad'] & djt['dc'], theta['kns']),
    ctrl.Rule(jt['ad'] & djt['dl'], theta['knr']),
    ctrl.Rule(jt['ad'] & djt['s'],  theta['l']),
    ctrl.Rule(jt['ad'] & djt['jl'], theta['krr']),
    ctrl.Rule(jt['ad'] & djt['jc'], theta['krs']),
    ctrl.Rule(jt['n']  & djt['s'],  theta['l']),
    ctrl.Rule(jt['aj'] & djt['dc'], theta['krs']),
    ctrl.Rule(jt['aj'] & djt['dl'], theta['krr']),
    ctrl.Rule(jt['aj'] & djt['s'],  theta['l']),
    ctrl.Rule(jt['aj'] & djt['jl'], theta['knr']),
    ctrl.Rule(jt['aj'] & djt['jc'], theta['kns']),
    ctrl.Rule(jt['tj'] & djt['dc'], theta['krt']),
    ctrl.Rule(jt['tj'] & djt['dl'], theta['krs']),
    ctrl.Rule(jt['tj'] & djt['s'],  theta['krr']),
    ctrl.Rule(jt['tj'] & djt['jl'], theta['l']),
    ctrl.Rule(jt['tj'] & djt['jc'], theta['knr']),
]

fuzzy_ctrl = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# ====== Load Model ======
model = tf.keras.models.load_model('D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/datasetbener3coba.keras')
input_size = (448, 256)

# ====== Webcam Start ======
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

prev_jt_cm = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    center_x = orig_w // 2

    # Preprocess
    input_frame = cv2.resize(frame, input_size)
    input_tensor = np.expand_dims(input_frame / 255.0, axis=0)

    # Predict
    pred = model.predict(input_tensor)[0]
    mask = np.argmax(pred, axis=-1).astype(np.uint8)
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Hitung Jarak ke Tepi Kiri (jt_cm)
    left_edge_pixels = np.where(mask_resized[:, :center_x] == 2)[1]
    jt_pixel = center_x - left_edge_pixels.min() if left_edge_pixels.size > 0 else center_x
    jt_cm = int(jt_pixel * 0.25)

    # Perubahan jarak Δjt
    djt_cm = jt_cm - prev_jt_cm if prev_jt_cm is not None else 0
    prev_jt_cm = jt_cm

    # Evaluasi Fuzzy
    try:
        fuzzy_sim.input['jt'] = np.clip(jt_cm, 0, 120)
        fuzzy_sim.input['djt'] = np.clip(djt_cm, -20, 20)
        fuzzy_sim.compute()
        theta_output = fuzzy_sim.output['theta']
    except Exception as e:
        print(f"[Fuzzy ERROR] jt={jt_cm}, djt={djt_cm} → {e}")
        theta_output = 0

    # Visualisasi hasil
    overlay = frame.copy()
    cv2.putText(overlay, f"Jarak Tepi Kiri: {jt_cm} cm", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(overlay, f"Delta Jarak: {djt_cm} cm", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(overlay, f"Theta Fuzzy: {theta_output:.1f} derajat", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # Segmentasi visual
    color_mask = np.zeros_like(frame)
    color_mask[mask_resized == 1] = [0, 255, 0]    # Jalan
    color_mask[mask_resized == 2] = [0, 0, 255]    # Tepi
    seg_overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)

    cv2.imshow("Segmentasi & Navigasi Fuzzy", seg_overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
