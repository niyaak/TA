import cv2
import numpy as np
import onnxruntime as ort
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import time
import os
from collections import deque

# ==============================================================================
# BAGIAN 1: LOGIKA FUZZY (Tidak ada perubahan)
# ==============================================================================
jt = ctrl.Antecedent(np.arange(0, 121, 1), 'jt')
djt = ctrl.Antecedent(np.arange(-20, 21, 1), 'djt')
theta = ctrl.Consequent(np.arange(-60, 61, 1), 'theta')
jt['td'] = fuzz.trapmf(jt.universe, [0, 0, 10, 20]); jt['ad'] = fuzz.trimf(jt.universe, [15, 22, 30]); jt['n']  = fuzz.trapmf(jt.universe, [20, 25, 35, 40]); jt['aj'] = fuzz.trimf(jt.universe, [30, 37, 45]); jt['tj'] = fuzz.trapmf(jt.universe, [40, 50, 120, 120])
djt['dc'] = fuzz.trapmf(djt.universe, [-20, -20, -12, -8]); djt['dl'] = fuzz.trimf(djt.universe, [-10, -6, -2]); djt['s'] = fuzz.trimf(djt.universe, [-3, 0, 3]); djt['jl'] = fuzz.trimf(djt.universe, [2, 6, 10]); djt['jc'] = fuzz.trapmf(djt.universe, [8, 12, 20, 20])
theta['krt'] = fuzz.trapmf(theta.universe, [-60, -60, -50, -40]); theta['krs'] = fuzz.trimf(theta.universe, [-45, -30, -15]); theta['krr'] = fuzz.trimf(theta.universe, [-20, -10, -2]); theta['l'] = fuzz.trimf(theta.universe, [-5, 0, 5]); theta['knr'] = fuzz.trimf(theta.universe, [2, 10, 20]); theta['kns'] = fuzz.trimf(theta.universe, [15, 30, 45]); theta['knt'] = fuzz.trapmf(theta.universe, [40, 50, 60, 60])
rules = [ctrl.Rule(jt['td']&djt['dc'],theta['krt']),ctrl.Rule(jt['td']&djt['dl'],theta['krs']),ctrl.Rule(jt['td']&djt['s'],theta['krr']),ctrl.Rule(jt['td']&djt['jl'],theta['l']),ctrl.Rule(jt['td']&djt['jc'],theta['knr']),ctrl.Rule(jt['ad']&djt['dc'],theta['krs']),ctrl.Rule(jt['ad']&djt['dl'],theta['krr']),ctrl.Rule(jt['ad']&djt['s'],theta['krr']),ctrl.Rule(jt['ad']&djt['jl'],theta['l']),ctrl.Rule(jt['ad']&djt['jc'],theta['knr']),ctrl.Rule(jt['n']&djt['dc'],theta['krr']),ctrl.Rule(jt['n']&djt['dl'],theta['krr']),ctrl.Rule(jt['n']&djt['s'],theta['l']),ctrl.Rule(jt['n']&djt['jl'],theta['knr']),ctrl.Rule(jt['n']&djt['jc'],theta['knr']),ctrl.Rule(jt['aj']&djt['dc'],theta['l']),ctrl.Rule(jt['aj']&djt['dl'],theta['l']),ctrl.Rule(jt['aj']&djt['s'],theta['knr']),ctrl.Rule(jt['aj']&djt['jl'],theta['kns']),ctrl.Rule(jt['aj']&djt['jc'],theta['knt']),ctrl.Rule(jt['tj']&djt['dc'],theta['krr']),ctrl.Rule(jt['tj']&djt['dl'],theta['l']),ctrl.Rule(jt['tj']&djt['s'],theta['knt']),ctrl.Rule(jt['tj']&djt['jl'],theta['knt']),ctrl.Rule(jt['tj']&djt['jc'],theta['knt'])]
fuzzy_ctrl = ctrl.ControlSystem(rules)
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# ==============================================================================
# BAGIAN 2: SETUP DAN KONFIGURASI
# ==============================================================================
ONNX_MODEL_PATH = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model2_onnx.onnx'
try:
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    _, height, width, _ = ort_session.get_inputs()[0].shape
    input_size = (width, height)
    print(f"Model ONNX berhasil dimuat. Input: {input_name}, Output: {output_name}, Size: {input_size}")
except Exception as e:
    print(f"Error memuat model ONNX: {e}"); exit()

INPUT_SOURCE = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/vidio/tes/democpt2.mp4'
cap = cv2.VideoCapture(INPUT_SOURCE)
if not cap.isOpened():
    print(f"Error: Tidak bisa membuka sumber video: {INPUT_SOURCE}"); exit()

prev_jt_cm_for_djt_calc = None
actual_previous_theta_output = 0.0
current_gps_navigation_mode = "LURUS"
theta_history = deque(maxlen=5)

# --- Konstanta Logika Adaptif (NILAI DIUBAH, MOHON VERIFIKASI) ---
# TODO: Ganti 280.0 dengan lebar lajur kendaraan Anda yang sebenarnya dalam cm.
DESIRED_LANE_WIDTH_CM = 280.0 
FUZZY_SYSTEM_NORMAL_JT_CM = 30.0 
SHARP_CURVE_COEFF_A_THRESHOLD = 0.0004 
SHARP_CURVE_ADJUSTMENT_FACTOR = 0.6 

# --- Konstanta Kalibrasi (NILAI DIUBAH, MOHON VERIFIKASI) ---
CM_PER_PIXEL = 0.12
MIN_ROAD_PIXELS_TO_OPERATE = 50
MAX_FRAMES_EDGE_LOST_FALLBACK = 3
Y_SAMPLING_START_FACTOR = 1.0
Y_SAMPLING_END_FACTOR = 0.5
# TODO: Y_REF_FOR_JT_FACTOR diubah ke 0.85 untuk stabilitas. 0.98 terlalu dekat ke bawah.
Y_REF_FOR_JT_FACTOR = 0.85
NUM_Y_WINDOWS_STRAIGHT = 3
NUM_EDGE_POINTS_CURVE = 7
TEPI_CLASS_VALUE = 2 

try: script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: script_dir = os.getcwd()
output_video_filename = "hasil_navigasi_final.mp4"
output_video_path = os.path.join(script_dir, output_video_filename)
video_writer = None; output_video_fps = 20.0

# --- Konstanta Visualisasi (sesuai kode asli Anda) ---
COLOR_SAMPLED_EDGE_POINTS = (255, 200, 100); COLOR_POINTS_FOR_FIT = (0, 255, 0); COLOR_FITTED_LINE = (150, 50, 0)
RADIUS_SAMPLED_POINTS = 3; RADIUS_POINTS_FOR_FIT = 5; THICKNESS_FITTED_LINE = 2
COLOR_TARGET_PATH = (0, 255, 112)

# ==============================================================================
# BAGIAN 3: FUNGSI BANTU
# ==============================================================================
def get_steering_decision_text(theta_value):
    if theta_value < -35: return "KANAN TAJAM"
    elif theta_value < -20: return "KANAN SEDANG"
    elif theta_value < -3: return "KANAN RINGAN"
    elif theta_value <= 3: return "LURUS"
    elif theta_value <= 20: return "KIRI RINGAN"
    elif theta_value <= 35: return "KIRI SEDANG"
    else: return "KIRI TAJAM"

def get_rightmost_edge_pixel_on_row(mask_segment, y_row, center_x):
    if y_row < 0 or y_row >= mask_segment.shape[0]: return None
    left_half_row = mask_segment[y_row, :center_x]
    edge_pixels_x = np.where(left_half_row == TEPI_CLASS_VALUE)[0]
    return edge_pixels_x.max() if edge_pixels_x.size > 0 else None

def fit_line_or_curve(points_x, points_y, mode="LURUS"):
    order = 1 if mode == "LURUS" else 2
    min_points = 2 if mode == "LURUS" else 3
    if len(points_x) >= min_points:
        try:
            return np.poly1d(np.polyfit(points_y, points_x, order))
        except Exception as e:
            print(f"Peringatan polyfit {mode}: {e}")
    return None

# ==============================================================================
# BAGIAN 4: MAIN LOOP
# ==============================================================================
while True:
    loop_start_time = time.time()
    ret, frame = cap.read()
    if not ret: print("Stream berakhir."); break

    orig_h, orig_w = frame.shape[:2]; center_x_frame = orig_w // 2

    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, output_video_fps, (orig_w, orig_h))
        print(f"Menyimpan video ke: {output_video_path}")
        
    # --- Inisialisasi variabel untuk loop ini ---
    raw_theta_output = actual_previous_theta_output
    status_info_for_decision = ""
    poly_func_for_vis = None
    target_path_poly_for_vis = None
    is_sharp_curve_detected = False
    current_target_offset_cm = DESIRED_LANE_WIDTH_CM / 2.0
    found_edge_for_fitting = False
    # Variabel untuk visualisasi asli Anda
    edge_points_x_for_fitting_vis, edge_points_y_for_fitting_vis = [], []
    show_jt_reference_marker = False
    reference_x_for_vis, vis_y_edge_for_marker = -1, -1
    display_jt_cm, display_djt_cm = "N/A", "N/A"
    
    # --- PRE-PROCESSING & PREDIKSI ONNX (SUDAH DIPERBAIKI) ---
    input_frame_resized = cv2.resize(frame, input_size)
    input_tensor = np.expand_dims(input_frame_resized, axis=0)
    
    try:
        ort_inputs = {input_name: input_tensor.astype(np.float32)}
        pred_onnx = ort_session.run([output_name], ort_inputs)[0]
        mask = np.argmax(pred_onnx[0], axis=-1).astype(np.uint8)
        mask_resized_to_orig = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(f"Error saat prediksi ONNX: {e}"); mask_resized_to_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)

    # --- LOGIKA INTI ADAPTIF ---
    if np.sum(mask_resized_to_orig == TEPI_CLASS_VALUE) < MIN_ROAD_PIXELS_TO_OPERATE:
        raw_theta_output = 0.0
        status_info_for_decision = "(TEPI SANGAT MINIM/HILANG!)"
    else:
        edge_points_x_to_fit, edge_points_y_to_fit = [], []
        y_start, y_end = int(orig_h * Y_SAMPLING_START_FACTOR), int(orig_h * Y_SAMPLING_END_FACTOR)
        
        # ... Logika fitting sama seperti sebelumnya ...
        poly_func = None
        if current_gps_navigation_mode == "LURUS":
            y_steps = np.linspace(y_end, y_start, NUM_Y_WINDOWS_STRAIGHT + 1, dtype=int)
            for i in range(NUM_Y_WINDOWS_STRAIGHT):
                win_y_top, win_y_bottom = y_steps[i], y_steps[i+1]
                y_mid = (win_y_top + win_y_bottom) // 2
                x_edge = get_rightmost_edge_pixel_on_row(mask_resized_to_orig, y_mid, center_x_frame)
                if x_edge is not None: edge_points_x_to_fit.append(x_edge); edge_points_y_to_fit.append(y_mid)
            poly_func = fit_line_or_curve(edge_points_x_to_fit, edge_points_y_to_fit, "LURUS")
        else:
            y_horizons = np.linspace(y_end, y_start, NUM_EDGE_POINTS_CURVE, dtype=int)
            for y_h in y_horizons:
                x_edge = get_rightmost_edge_pixel_on_row(mask_resized_to_orig, y_h, center_x_frame)
                if x_edge is not None: edge_points_x_to_fit.append(x_edge); edge_points_y_to_fit.append(y_h)
            poly_func = fit_line_or_curve(edge_points_x_to_fit, edge_points_y_to_fit, "BELOK")
            
        if poly_func:
            found_edge_for_fitting = True
            poly_func_for_vis = poly_func
            edge_points_x_for_fitting_vis = list(edge_points_x_to_fit)
            edge_points_y_for_fitting_vis = list(edge_points_y_to_fit)

            if current_gps_navigation_mode == "BELOK" and len(poly_func.coeffs) > 1:
                if abs(poly_func.coeffs[0]) > SHARP_CURVE_COEFF_A_THRESHOLD:
                    is_sharp_curve_detected = True
                    current_target_offset_cm *= SHARP_CURVE_ADJUSTMENT_FACTOR
            
            target_offset_pixels = current_target_offset_cm / CM_PER_PIXEL
            target_path_poly_for_vis = poly_func + target_offset_pixels
            
            try:
                y_ref = int(orig_h * Y_REF_FOR_JT_FACTOR)
                ref_x = int(poly_func(y_ref))
                if not (0 <= ref_x < orig_w): raise ValueError("Ref x out of screen bounds")
                
                reference_x_for_vis = ref_x
                vis_y_edge_for_marker = y_ref
                show_jt_reference_marker = True

                measured_jt = (center_x_frame - ref_x) * CM_PER_PIXEL
                djt = (measured_jt - prev_jt_cm_for_djt_calc) if prev_jt_cm_for_djt_calc is not None else 0
                prev_jt_cm_for_djt_calc = measured_jt
                
                display_jt_cm = f"{measured_jt:.1f}"
                display_djt_cm = f"{djt:.1f}"

                offset_correction = current_target_offset_cm - FUZZY_SYSTEM_NORMAL_JT_CM
                jt_for_fuzzy = measured_jt - offset_correction
                
                fuzzy_sim.input['jt'] = np.clip(jt_for_fuzzy, 0, 120)
                fuzzy_sim.input['djt'] = np.clip(djt, -20, 20)
                fuzzy_sim.compute()
                raw_theta_output = fuzzy_sim.output['theta']
            except Exception as e:
                print(f"Error saat proses fuzzy: {e}"); prev_jt_cm_for_djt_calc = None

    # --- SMOOTHING & KEPUTUSAN FINAL ---
    theta_history.append(raw_theta_output)
    current_theta_output = np.mean(theta_history)
    core_decision_text = get_steering_decision_text(current_theta_output)
    actual_previous_theta_output = current_theta_output
    
    # ==============================================================================
    # BAGIAN 5: VISUALISASI (SESUAI PERMINTAAN ANDA)
    # ==============================================================================
    overlay = frame.copy()
    
    # Tampilkan teks info
    y_offset = 30; info_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, f"JT (Actual): {display_jt_cm} cm", (10, y_offset), info_font, 0.6, (0, 255, 255), 2); y_offset += 25
    cv2.putText(overlay, f"dJT: {display_djt_cm} cm/f", (10, y_offset), info_font, 0.6, (0, 255, 255), 2); y_offset += 25
    cv2.putText(overlay, f"Target Offset: {current_target_offset_cm:.1f} cm", (10, y_offset), info_font, 0.6, COLOR_TARGET_PATH, 2); y_offset += 25
    cv2.putText(overlay, f"Theta (Sm): {current_theta_output:.1f} deg", (10, y_offset), info_font, 0.6, (0, 255, 255), 2); y_offset += 25
    fps = 1.0 / (time.time() - loop_start_time)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, y_offset), info_font, 0.6, (200, 200, 0), 1)

    # Tampilkan keputusan utama
    if is_sharp_curve_detected: core_decision_text += " (TIKUNGAN TAJAM!)"
    cv2.putText(overlay, f"KEPUTUSAN: {core_decision_text}", (10, y_offset + 30), info_font, 0.7, (0, 255, 0), 2)
    
    # Buat color mask untuk hasil segmentasi
    color_mask = np.zeros_like(frame); color_mask[mask_resized_to_orig == TEPI_CLASS_VALUE] = [0, 0, 255]
    
    # Gambar titik-titik hijau yang digunakan untuk fitting
    if len(edge_points_x_for_fitting_vis) > 0:
        for i in range(len(edge_points_x_for_fitting_vis)):
            cv2.circle(overlay, (edge_points_x_for_fitting_vis[i], edge_points_y_for_fitting_vis[i]), RADIUS_POINTS_FOR_FIT, COLOR_POINTS_FOR_FIT, -1)
    
    # Gambar garis hasil fitting
    if found_edge_for_fitting and poly_func_for_vis is not None:
        y_draw_start = int(orig_h * 0.98); y_draw_end = int(orig_h * Y_SAMPLING_END_FACTOR)
        prev_x_p, prev_y_p = -1, -1
        for y_p in range(y_draw_start, y_draw_end -1 , -1): 
            try:
                x_p = int(poly_func_for_vis(y_p))
                if 0 <= x_p < orig_w:
                    if prev_x_p != -1: cv2.line(overlay, (prev_x_p, prev_y_p), (x_p, y_p), COLOR_FITTED_LINE, THICKNESS_FITTED_LINE)
                    prev_x_p, prev_y_p = x_p, y_p
                else: prev_x_p, prev_y_p = -1, -1
            except: pass 

    # Gambar garis jalur target
    if target_path_poly_for_vis is not None:
        y_draw_start = int(orig_h * 0.98); y_draw_end = int(orig_h * Y_SAMPLING_END_FACTOR)
        prev_x_p, prev_y_p = -1, -1
        for y_p in range(y_draw_start, y_draw_end - 1, -1):
            try:
                x_p = int(target_path_poly_for_vis(y_p))
                if 0 <= x_p < orig_w:
                    if prev_x_p != -1: cv2.line(overlay, (prev_x_p, prev_y_p), (x_p, y_p), COLOR_TARGET_PATH, THICKNESS_FITTED_LINE)
                    prev_x_p, prev_y_p = x_p, y_p
                else: prev_x_p, prev_y_p = -1, -1
            except: pass
    
    # Gambar marker referensi JT
    if show_jt_reference_marker:
        cv2.circle(overlay, (reference_x_for_vis, vis_y_edge_for_marker), 7, (0, 255, 255), -1) 
        cv2.line(overlay, (center_x_frame, vis_y_edge_for_marker), (reference_x_for_vis, vis_y_edge_for_marker), (255,255,0), 1)
    
    # Gabungkan overlay dengan color mask
    final_frame = cv2.addWeighted(overlay, 1, color_mask, 0.3, 0)
    
    cv2.imshow("Navigasi ONNX Adaptif", final_frame)
    if video_writer: video_writer.write(final_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('l'): current_gps_navigation_mode = "LURUS"
    elif key == ord('b'): current_gps_navigation_mode = "BELOK"

# ==============================================================================
# BAGIAN 6: CLEANUP
# ==============================================================================
cap.release()
if video_writer: video_writer.release()
cv2.destroyAllWindows()