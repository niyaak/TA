import cv2
import numpy as np
import tensorflow as tf
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import time # Import modul time

# ====== Fuzzy Variable Definitions ======
# ... (Definisi fuzzy tetap sama seperti sebelumnya) ...
jt = ctrl.Antecedent(np.arange(0, 121, 1), 'jt')
djt = ctrl.Antecedent(np.arange(-20, 21, 1), 'djt')
theta = ctrl.Consequent(np.arange(-60, 61, 1), 'theta')

jt['td'] = fuzz.trapmf(jt.universe, [0, 0, 5, 10])
jt['ad'] = fuzz.trimf(jt.universe, [5, 15, 25])
jt['n']  = fuzz.trapmf(jt.universe, [20, 25, 35, 40])
jt['aj'] = fuzz.trimf(jt.universe, [35, 45, 55])
jt['tj'] = fuzz.trapmf(jt.universe, [50, 60, 120, 120])

djt['dc'] = fuzz.trapmf(djt.universe, [-20, -20, -12, -8]); djt['dl'] = fuzz.trimf(djt.universe, [-10, -6, -2]); djt['s'] = fuzz.trimf(djt.universe, [-3, 0, 3]); djt['jl'] = fuzz.trimf(djt.universe, [2, 6, 10]); djt['jc'] = fuzz.trapmf(djt.universe, [8, 12, 20, 20])
theta['krt'] = fuzz.trapmf(theta.universe, [-60, -60, -50, -40]); theta['krs'] = fuzz.trimf(theta.universe, [-45, -30, -15]); theta['krr'] = fuzz.trimf(theta.universe, [-20, -10, -2]); theta['l'] = fuzz.trimf(theta.universe, [-5, 0, 5]); theta['knr'] = fuzz.trimf(theta.universe, [2, 10, 20]); theta['kns'] = fuzz.trimf(theta.universe, [15, 30, 45]); theta['knt'] = fuzz.trapmf(theta.universe, [40, 50, 60, 60])

rules = [
    ctrl.Rule(jt['td'] & djt['dc'], theta['krt']), ctrl.Rule(jt['td'] & djt['dl'], theta['krs']),
    ctrl.Rule(jt['td'] & djt['s'], theta['krr']), ctrl.Rule(jt['td'] & djt['jl'], theta['l']),
    ctrl.Rule(jt['td'] & djt['jc'], theta['knr']),
    ctrl.Rule(jt['ad'] & djt['dc'], theta['krs']), ctrl.Rule(jt['ad'] & djt['dl'], theta['krr']),
    ctrl.Rule(jt['ad'] & djt['s'], theta['krr']), ctrl.Rule(jt['ad'] & djt['jl'], theta['l']),
    ctrl.Rule(jt['ad'] & djt['jc'], theta['knr']),
    ctrl.Rule(jt['n'] & djt['dc'], theta['krr']), ctrl.Rule(jt['n'] & djt['dl'], theta['krr']),
    ctrl.Rule(jt['n'] & djt['s'], theta['l']), ctrl.Rule(jt['n'] & djt['jl'], theta['knr']),
    ctrl.Rule(jt['n'] & djt['jc'], theta['knr']),
    ctrl.Rule(jt['aj'] & djt['dc'], theta['l']), ctrl.Rule(jt['aj'] & djt['dl'], theta['l']),
    ctrl.Rule(jt['aj'] & djt['s'], theta['knr']), ctrl.Rule(jt['aj'] & djt['jl'], theta['kns']),
    ctrl.Rule(jt['aj'] & djt['jc'], theta['knt']),
    ctrl.Rule(jt['tj'] & djt['dc'], theta['krr']), ctrl.Rule(jt['tj'] & djt['dl'], theta['l']),
    ctrl.Rule(jt['tj'] & djt['s'], theta['knt']), ctrl.Rule(jt['tj'] & djt['jl'], theta['knt']),
    ctrl.Rule(jt['tj'] & djt['jc'], theta['knt']),
]
fuzzy_ctrl = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

MODEL_PATH = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/datasetbener3coba.keras' # Sesuaikan path ini
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
input_size = (448, 256)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera/video.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variabel state sebelumnya
prev_jt_cm_for_djt_calc = None
actual_previous_theta_output = 0.0
actual_previous_core_decision_text = "LURUS (Inisialisasi)"

# Konstanta Kalibrasi & Logika
CM_PER_PIXEL = 0.12
ROAD_DETECTION_THRESHOLD_PERCENT = 80.0

# Variabel untuk perhitungan FPS
prev_frame_time = time.time()
new_frame_time = time.time()

def get_steering_decision_text(theta_value):
    if theta_value < -35: return "KANAN TAJAM"
    elif theta_value < -20: return "KANAN SEDANG"
    elif theta_value < -3: return "KANAN RINGAN"
    elif theta_value <= 3: return "LURUS"
    elif theta_value <= 20: return "KIRI RINGAN"
    elif theta_value <= 35: return "KIRI SEDANG"
    else: return "KIRI TAJAM"

while True:
    loop_start_time = time.time() # Waktu awal loop untuk FPS
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame, stream berakhir.")
        break

    orig_h, orig_w = frame.shape[:2]
    center_x_frame = orig_w // 2

    current_theta_output = 0.0
    core_decision_for_this_frame = "N/A" 
    status_info_for_decision = ""       
    display_jt_cm = "N/A"
    display_djt_cm = "N/A"
    show_edge_marker = False
    min_x_coord_of_left_edge_vis = -1
    vis_y_edge_vis = -1
    road_percentage_display = 0.0
    inference_time_ms = 0.0 # Inisialisasi inference time

    input_frame_resized = cv2.resize(frame, input_size)
    input_tensor = np.expand_dims(input_frame_resized / 255.0, axis=0)
    
    mask_resized_to_orig = np.zeros((orig_h, orig_w), dtype=np.uint8) # Default dummy mask

    try:
        start_time_inference = time.time()
        pred = model.predict(input_tensor, verbose=0)[0]
        end_time_inference = time.time()
        inference_time_ms = (end_time_inference - start_time_inference) * 1000

        mask = np.argmax(pred, axis=-1).astype(np.uint8)
        mask_resized_to_orig = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(f"Error saat prediksi model: {e}")
        inference_time_ms = -1.0 # Menandakan error
        current_theta_output = actual_previous_theta_output
        core_decision_for_this_frame = actual_previous_core_decision_text
        status_info_for_decision = "(Prediksi Gagal)"
        # mask_resized_to_orig sudah diinisialisasi sebagai dummy mask


    # 1. Deteksi Jalan di ROI
    roi_y_start = int(orig_h * 0.8)
    roi_y_end = int(orig_h * 1)
    roi_x_start = int(center_x_frame - (orig_w * 0.20))
    roi_x_end = int(center_x_frame + (orig_w * 0.20))
    
    road_roi_segment = mask_resized_to_orig[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
    
    is_road_adequately_detected = False
    if road_roi_segment.size > 0:
        road_pixel_count = np.sum(road_roi_segment == 1)
        total_pixels_in_roi = road_roi_segment.size
        road_percentage_display = (road_pixel_count / total_pixels_in_roi) * 100
        if road_percentage_display >= ROAD_DETECTION_THRESHOLD_PERCENT:
            is_road_adequately_detected = True

    if not is_road_adequately_detected:
        current_theta_output = 0.0
        core_decision_for_this_frame = "BERHENTI"
        status_info_for_decision = "(TIDAK ADA JALAN)"
        # display_jt_cm, display_djt_cm tetap "N/A"
        show_edge_marker = False
    else:
        search_area_for_edge = mask_resized_to_orig[roi_y_start:roi_y_end, :center_x_frame]
        edge_pixels_in_search_area = np.where(search_area_for_edge == 2)

        if edge_pixels_in_search_area[1].size > 0:
            x_coords_in_search_area = edge_pixels_in_search_area[1]
            y_coords_in_search_area_relative_to_roi = edge_pixels_in_search_area[0]
            
            min_x_val_abs = x_coords_in_search_area.min()
            min_x_coord_of_left_edge_vis = min_x_val_abs

            y_indices_at_min_x = y_coords_in_search_area_relative_to_roi[x_coords_in_search_area == min_x_val_abs]
            if y_indices_at_min_x.size > 0:
                vis_y_edge_vis = int(np.mean(y_indices_at_min_x)) + roi_y_start
            
            jt_pixel_val = center_x_frame - min_x_val_abs
            jt_cm_val = int(jt_pixel_val * CM_PER_PIXEL)
            
            djt_cm_val = 0
            if prev_jt_cm_for_djt_calc is not None:
                djt_cm_val = jt_cm_val - prev_jt_cm_for_djt_calc
            prev_jt_cm_for_djt_calc = jt_cm_val

            try:
                clipped_jt_cm = np.clip(jt_cm_val, jt.universe.min(), jt.universe.max())
                clipped_djt_cm = np.clip(djt_cm_val, djt.universe.min(), djt.universe.max())
                fuzzy_sim.input['jt'] = clipped_jt_cm
                fuzzy_sim.input['djt'] = clipped_djt_cm
                fuzzy_sim.compute()
                current_theta_output = fuzzy_sim.output['theta']
                core_decision_for_this_frame = get_steering_decision_text(current_theta_output)
                status_info_for_decision = ""
            except Exception as e:
                print(f"[Fuzzy ERROR] jt={jt_cm_val}, djt={djt_cm_val} -> {e}")
                current_theta_output = actual_previous_theta_output
                core_decision_for_this_frame = actual_previous_core_decision_text
                status_info_for_decision = "(Fuzzy Error)"

            display_jt_cm = f"{jt_cm_val:.1f}"
            display_djt_cm = f"{djt_cm_val:.1f}"
            show_edge_marker = True
        else:
            current_theta_output = actual_previous_theta_output
            core_decision_for_this_frame = actual_previous_core_decision_text
            status_info_for_decision = "(TEPI KIRI HILANG DI ROI)"
            # display_jt_cm, display_djt_cm tetap "N/A"
            show_edge_marker = False
            min_x_coord_of_left_edge_vis = -1

    current_decision_text = core_decision_for_this_frame
    if status_info_for_decision:
        current_decision_text += f" {status_info_for_decision}"

    actual_previous_theta_output = current_theta_output
    actual_previous_core_decision_text = core_decision_for_this_frame

    # ====== Visualisasi ======
    overlay = frame.copy()
    cv2.line(overlay, (center_x_frame, 0), (center_x_frame, orig_h), (255, 0, 0), 1)
    cv2.rectangle(overlay, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0,165,255), 1)
    cv2.putText(overlay, f"ROI Cek Jalan ({road_percentage_display:.0f}%)", (roi_x_start, roi_y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255),1)

    y_offset = 30; info_text_size = 0.6; info_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, f"JT: {display_jt_cm} cm", (10, y_offset), info_font, info_text_size, (0, 255, 255), 2); y_offset += 25
    cv2.putText(overlay, f"dJT: {display_djt_cm} cm/f", (10, y_offset), info_font, info_text_size, (0, 255, 255), 2); y_offset += 25
    cv2.putText(overlay, f"Theta: {current_theta_output:.1f} deg", (10, y_offset), info_font, info_text_size, (0, 255, 255), 2); y_offset += 30
    
    keputusan_color = (0, 255, 0)
    if "BERHENTI" in core_decision_for_this_frame: keputusan_color = (0, 0, 255)
    elif status_info_for_decision : keputusan_color = (0, 255, 255)
    cv2.putText(overlay, f"KEPUTUSAN: {current_decision_text}", (10, y_offset), info_font, 0.7, keputusan_color, 2); y_offset += 25 # Sedikit ruang tambahan

    # Tampilkan Inference Time dan FPS
    cv2.putText(overlay, f"Infer. Time: {inference_time_ms:.1f} ms", (10, y_offset), info_font, info_text_size-0.1, (200, 200, 0), 1); y_offset += 20
    
    # Hitung FPS di akhir pemrosesan frame
    new_frame_time = time.time() # Waktu akhir loop
    time_diff_for_fps = new_frame_time - loop_start_time # Gunakan loop_start_time yang diambil di awal loop
    if time_diff_for_fps > 0:
        fps = 1.0 / time_diff_for_fps
    else:
        fps = 0 
    # prev_frame_time = new_frame_time # Tidak diperlukan jika menggunakan loop_start_time
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, y_offset), info_font, info_text_size-0.1, (200, 200, 0), 1); y_offset += 20


    color_mask = np.zeros_like(frame); color_mask[mask_resized_to_orig == 1] = [0, 255, 0]; color_mask[mask_resized_to_orig == 2] = [0, 0, 255]

    if show_edge_marker and min_x_coord_of_left_edge_vis != -1 and vis_y_edge_vis != -1:
        cv2.circle(overlay, (min_x_coord_of_left_edge_vis, vis_y_edge_vis), 7, (0, 255, 255), -1)
        cv2.line(overlay, (center_x_frame, vis_y_edge_vis), (min_x_coord_of_left_edge_vis, vis_y_edge_vis), (255,255,0), 1)

    seg_overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
    cv2.imshow("Segmentasi & Navigasi Fuzzy", seg_overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()