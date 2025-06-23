import cv2
import numpy as np
import tensorflow as tf
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import time
import os
from collections import deque
import openpyxl # <-- Tambahkan import ini
from datetime import datetime # <-- Tambahkan import ini

# ====== Fuzzy Variable Definitions ======
jt = ctrl.Antecedent(np.arange(0, 121, 1), 'jt')
djt = ctrl.Antecedent(np.arange(-20, 21, 1), 'djt')
theta = ctrl.Consequent(np.arange(-60, 61, 1), 'theta')

# Membership Functions for JT (Jarak Tepi)
jt['td'] = fuzz.trapmf(jt.universe, [0, 0, 10, 20])
jt['ad'] = fuzz.trimf(jt.universe, [15, 22, 30])
jt['n']  = fuzz.trapmf(jt.universe, [20, 25, 35, 40])
jt['aj'] = fuzz.trimf(jt.universe, [30, 37, 45])
jt['tj'] = fuzz.trapmf(jt.universe, [40, 50, 120, 120])

# Membership Functions for dJT & Theta
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

#MODEL_PATH = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/effnet.keras'
#MODEL_PATH = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/basic1024.keras'
#MODEL_PATH = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/basic256.keras'
#MODEL_PATH = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/basic512.keras'
MODEL_PATH = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/mobileunet.keras'

#MODEL_PATH = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/mobilenet2.keras'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model {MODEL_PATH} berhasil dimuat.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

input_size = (384, 224 )
#input_size = (448, 256 )

INPUT_SOURCE = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/vidio/tes/democpt2.mp4'
#INPUT_SOURCE = 0

cap = cv2.VideoCapture(INPUT_SOURCE)
if not cap.isOpened():
    print(f"Error: Tidak bisa membuka sumber video/kamera: {INPUT_SOURCE}")
    exit()

if isinstance(INPUT_SOURCE, int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# --- Variabel state & smoothing ---
prev_jt_cm_for_djt_calc = None
actual_previous_theta_output = 0.0
actual_previous_core_decision_text = "LURUS (Inisialisasi)"
frames_edge_lost_counter = 0
current_gps_navigation_mode = "LURUS"
THETA_SMOOTHING_WINDOW_SIZE = 5
theta_history = deque(maxlen=THETA_SMOOTHING_WINDOW_SIZE)

# Variabel untuk validasi konsistensi tepi jalan
last_stable_reference_x = None
outlier_detection_counter = 0
OUTLIER_X_JUMP_THRESHOLD_PX = 60
OUTLIER_CONFIRMATION_FRAMES = 4

# --- Konstanta untuk Jalur Target & Adaptasi Tikungan ---
DESIRED_LANE_WIDTH_CM = 123.0
FUZZY_SYSTEM_NORMAL_JT_CM = 30.0
SHARP_CURVE_COEFF_A_THRESHOLD = 0.0004
SHARP_CURVE_ADJUSTMENT_FACTOR = 0.6

# --- Konstanta Kalibrasi & Logika ---
CM_PER_PIXEL = 0.26
MIN_ROAD_PIXELS_TO_OPERATE = 50
MAX_FRAMES_EDGE_LOST_FALLBACK = 3
Y_SAMPLING_START_FACTOR = 1
Y_SAMPLING_END_FACTOR = 0.55
Y_REF_FOR_JT_FACTOR = 0.95
NUM_Y_WINDOWS_STRAIGHT = 3
NUM_EDGE_POINTS_CURVE = 7
TEPI_CLASS_VALUE = 2

# --- NEW: Folder & Penamaan File Output Setup ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

# Buat folder output jika belum ada
log_output_dir = os.path.join(script_dir, "hasil_log")
video_output_dir = os.path.join(script_dir, "hasilvidio")
os.makedirs(log_output_dir, exist_ok=True)
os.makedirs(video_output_dir, exist_ok=True)

# Buat nama file unik berbasis timestamp
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"log_data_{timestamp_str}.xlsx"
output_video_overlay_filename = f"output_overlay_{timestamp_str}.mp4"
output_video_original_filename = f"output_original_{timestamp_str}.mp4"

log_filepath = os.path.join(log_output_dir, log_filename)
output_video_overlay_path = os.path.join(video_output_dir, output_video_overlay_filename)
output_video_original_path = os.path.join(video_output_dir, output_video_original_filename)

# --- NEW: Inisialisasi Logging Excel ---
workbook = openpyxl.Workbook()
sheet = workbook.active
sheet.title = "Frame Data Log"
# Tulis header ke file excel
excel_headers = [
    "Timestamp", "Frame", "FPS", "Infer_Time_ms", "JT_cm", "dJT_cm_per_frame",
    "JT_for_Fuzzy", "Theta_Raw", "Theta_Smoothed", "Keputusan_Inti", "Status_Info", "Keputusan_Final"
]
sheet.append(excel_headers)
print(f"Logging data ke: {log_filepath}")


# --- Video Output Setup ---
video_writer_overlay = None
video_writer_original = None # Writer baru untuk video original
output_video_fps = 10

# --- Konstanta Visualisasi ---
COLOR_SAMPLED_EDGE_POINTS = (255, 200, 100); COLOR_POINTS_FOR_FIT = (0, 255, 0); COLOR_FITTED_LINE = (150, 50, 0)
RADIUS_SAMPLED_POINTS = 3; RADIUS_POINTS_FOR_FIT = 5; THICKNESS_FITTED_LINE = 2
COLOR_TARGET_PATH = (0, 255, 112)

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
    if edge_pixels_x.size > 0: return edge_pixels_x.max()
    return None

def fit_line_or_curve(points_x, points_y, mode="LURUS"):
    order = 1 if mode == "LURUS" else 2; min_points = 2 if mode == "LURUS" else 3
    if len(points_x) >= min_points:
        try:
            if order > 0 and len(set(points_y)) < order + 1 and not (order == 1 and len(set(points_y)) >=1): return None
            coeffs = np.polyfit(points_y, points_x, order); return np.poly1d(coeffs)
        except Exception as e: print(f"Peringatan polyfit {mode}: {e}"); return None
    return None

frame_count = 0 # Counter untuk frame
while True:
    loop_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame, stream berakhir."); break
    
    frame_count += 1
    orig_h, orig_w = frame.shape[:2]
    center_x_frame = orig_w // 2

    # Inisialisasi Video Writers saat frame pertama didapat
    if video_writer_overlay is None:
        frame_size_out = (orig_w, orig_h); fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Writer untuk video dengan overlay
        video_writer_overlay = cv2.VideoWriter(output_video_overlay_path, fourcc, output_video_fps, frame_size_out)
        if not video_writer_overlay.isOpened(): 
            print(f"Error: Gagal VideoWriter (Overlay): {output_video_overlay_path}"); video_writer_overlay = None
        else: 
            print(f"Menyimpan video overlay ke: {output_video_overlay_path} @{output_video_fps} FPS")

        # Writer untuk video original
        video_writer_original = cv2.VideoWriter(output_video_original_path, fourcc, output_video_fps, frame_size_out)
        if not video_writer_original.isOpened():
             print(f"Error: Gagal VideoWriter (Original): {output_video_original_path}"); video_writer_original = None
        else:
            print(f"Menyimpan video original ke: {output_video_original_path} @{output_video_fps} FPS")


    # --- Inisialisasi variabel untuk frame ini ---
    raw_theta_output = actual_previous_theta_output
    core_decision_for_this_frame = actual_previous_core_decision_text
    status_info_for_decision = ""
    display_jt_cm = "N/A"; display_djt_cm = "N/A"
    jt_for_fuzzy_input_log = None; djt_cm_val_log = None; measured_jt_cm_log = None # Variabel log
    show_jt_reference_marker = False
    reference_x_for_vis = -1; vis_y_edge_for_marker = -1
    inference_time_ms = 0.0
    poly_func_for_vis = None
    edge_points_x_for_fitting_vis = []; edge_points_y_for_fitting_vis = []
    
    is_sharp_curve_detected = False
    target_path_poly_for_vis = None
    current_target_offset_cm = DESIRED_LANE_WIDTH_CM / 2.0

    mask_resized_to_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
    input_frame_resized = cv2.resize(frame, input_size)
    input_tensor = np.expand_dims(input_frame_resized / 255.0, axis=0)

    try:
        start_time_inference = time.time()
        pred = model.predict(input_tensor, verbose=0)[0]
        end_time_inference = time.time()
        inference_time_ms = (end_time_inference - start_time_inference) * 1000
        mask = np.argmax(pred, axis=-1).astype(np.uint8)
        mask_resized_to_orig = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print(f"Error saat prediksi model: {e}"); inference_time_ms = -1.0
        status_info_for_decision = "(Prediksi Gagal)"

    if np.sum(mask_resized_to_orig == TEPI_CLASS_VALUE) < MIN_ROAD_PIXELS_TO_OPERATE:
        raw_theta_output = 0.0; core_decision_for_this_frame = "LURUS"
        status_info_for_decision = "(TEPI SANGAT MINIM/HILANG!)"
        frames_edge_lost_counter = 0; prev_jt_cm_for_djt_calc = None
    else:
        edge_points_x_to_fit = []; edge_points_y_to_fit = []
        poly_func = None; found_edge_for_fitting = False
        y_sampling_start_abs = int(orig_h * Y_SAMPLING_START_FACTOR)
        y_sampling_end_abs = int(orig_h * Y_SAMPLING_END_FACTOR)

        if y_sampling_start_abs > y_sampling_end_abs:
            if current_gps_navigation_mode == "LURUS":
                num_windows = NUM_Y_WINDOWS_STRAIGHT; y_window_boundaries = np.linspace(y_sampling_end_abs, y_sampling_start_abs, num_windows + 1, dtype=int)
                for i in range(num_windows):
                    win_y_top = y_window_boundaries[i]; win_y_bottom = y_window_boundaries[i+1]
                    if win_y_bottom < win_y_top: continue
                    best_x_in_window = -1; y_for_best_x_in_window = -1
                    for y_h in range(win_y_bottom, win_y_top - 1, -1):
                        if y_h < 0 or y_h >= orig_h: continue
                        x_edge = get_rightmost_edge_pixel_on_row(mask_resized_to_orig, y_h, center_x_frame)
                        if x_edge is not None:
                            if x_edge > best_x_in_window: best_x_in_window = x_edge; y_for_best_x_in_window = y_h
                    if best_x_in_window != -1: edge_points_x_to_fit.append(best_x_in_window); edge_points_y_to_fit.append(y_for_best_x_in_window)
                if len(edge_points_x_to_fit) >= 2: poly_func = fit_line_or_curve(edge_points_x_to_fit, edge_points_y_to_fit, "LURUS")
            elif "BELOK" in current_gps_navigation_mode:
                y_horizons = np.linspace(y_sampling_start_abs, y_sampling_end_abs, NUM_EDGE_POINTS_CURVE, dtype=int)
                for y_h in y_horizons:
                    x_edge = get_rightmost_edge_pixel_on_row(mask_resized_to_orig, y_h, center_x_frame)
                    if x_edge is not None: edge_points_x_to_fit.append(x_edge); edge_points_y_to_fit.append(y_h)
                if len(edge_points_x_to_fit) >= 3: poly_func = fit_line_or_curve(edge_points_x_to_fit, edge_points_y_to_fit, "BELOK")

        if poly_func is not None:
            found_edge_for_fitting = True
            frames_edge_lost_counter = 0
            poly_func_for_vis = poly_func
            edge_points_x_for_fitting_vis = list(edge_points_x_to_fit)
            edge_points_y_for_fitting_vis = list(edge_points_y_to_fit)
            y_ref_for_jt_abs = int(orig_h * Y_REF_FOR_JT_FACTOR)

            is_detection_valid_this_frame = True
            try:
                potential_reference_x = int(poly_func(y_ref_for_jt_abs))
                if not (0 <= potential_reference_x < center_x_frame):
                    raise ValueError("Ref x (potensial) di luar batas valid.")

                if last_stable_reference_x is not None:
                    x_difference = abs(potential_reference_x - last_stable_reference_x)
                    
                    if x_difference > OUTLIER_X_JUMP_THRESHOLD_PX:
                        outlier_detection_counter += 1
                        if outlier_detection_counter < OUTLIER_CONFIRMATION_FRAMES:
                            is_detection_valid_this_frame = False
                            status_info_for_decision = f"(Anomali Tepi! Menunggu {outlier_detection_counter}/{OUTLIER_CONFIRMATION_FRAMES})"
                        else:
                            last_stable_reference_x = potential_reference_x
                            outlier_detection_counter = 0
                    else:
                        last_stable_reference_x = potential_reference_x
                        outlier_detection_counter = 0
                else:
                    last_stable_reference_x = potential_reference_x
                    outlier_detection_counter = 0

            except Exception as e:
                is_detection_valid_this_frame = False
                status_info_for_decision = "(Error Validasi Tepi)"
            
            if is_detection_valid_this_frame:
                if current_gps_navigation_mode == "BELOK" and len(poly_func.coeffs) == 3:
                    coeff_a = poly_func.coeffs[0]
                    if abs(coeff_a) > SHARP_CURVE_COEFF_A_THRESHOLD:
                        is_sharp_curve_detected = True
                        current_target_offset_cm = (DESIRED_LANE_WIDTH_CM / 2.0) * SHARP_CURVE_ADJUSTMENT_FACTOR

                target_offset_pixels = current_target_offset_cm / CM_PER_PIXEL
                target_path_poly_for_vis = poly_func + target_offset_pixels

                try:
                    reference_x_abs = last_stable_reference_x
                    reference_x_for_vis = reference_x_abs
                    vis_y_edge_for_marker = y_ref_for_jt_abs
                    
                    measured_jt_cm = (center_x_frame - reference_x_abs) * CM_PER_PIXEL
                    measured_jt_cm_log = measured_jt_cm # Simpan untuk log
                    djt_cm_val = 0
                    if prev_jt_cm_for_djt_calc is not None:
                        djt_cm_val = measured_jt_cm - prev_jt_cm_for_djt_calc
                    
                    djt_cm_val_log = djt_cm_val # Simpan untuk log
                    prev_jt_cm_for_djt_calc = measured_jt_cm

                    offset_correction = current_target_offset_cm - FUZZY_SYSTEM_NORMAL_JT_CM
                    jt_for_fuzzy_input = measured_jt_cm - offset_correction
                    jt_for_fuzzy_input_log = jt_for_fuzzy_input # Simpan untuk log

                    clipped_jt_cm = np.clip(jt_for_fuzzy_input, jt.universe.min(), jt.universe.max())
                    clipped_djt_cm = np.clip(djt_cm_val, djt.universe.min(), djt.universe.max())

                    fuzzy_sim.input['jt'] = clipped_jt_cm
                    fuzzy_sim.input['djt'] = clipped_djt_cm
                    fuzzy_sim.compute()
                    raw_theta_output = fuzzy_sim.output['theta']

                    if not status_info_for_decision:
                        status_info_for_decision = f"(Ref: {'Garis(F)' if current_gps_navigation_mode == 'LURUS' else 'Kurva'})"
                    if is_sharp_curve_detected:
                        status_info_for_decision += " (Tikungan Tajam!)"
                    display_jt_cm = f"{measured_jt_cm:.1f}"
                    display_djt_cm = f"{djt_cm_val:.1f}"
                    show_jt_reference_marker = True

                except Exception as e:
                    print(f"Error saat proses fitting/fuzzy: {e}")
                    status_info_for_decision = "(Error Proses Tepi)"
                    prev_jt_cm_for_djt_calc = None
                    is_detection_valid_this_frame = False
            
        if not found_edge_for_fitting:
            last_stable_reference_x = None
            outlier_detection_counter = 0
            
            frames_edge_lost_counter += 1
            if frames_edge_lost_counter <= MAX_FRAMES_EDGE_LOST_FALLBACK:
                status_info_for_decision = f"(TEPI HILANG SEMENTARA ({frames_edge_lost_counter}))"
            else:
                raw_theta_output = 0.0; core_decision_for_this_frame = "LURUS"
                status_info_for_decision = "(NAVIGASI PETA (GPS) AKTIF)"
            prev_jt_cm_for_djt_calc = None; display_jt_cm = "N/A"; display_djt_cm = "N/A (Tepi Hilang)"
            show_jt_reference_marker = False; reference_x_for_vis = -1
            edge_points_x_for_fitting_vis = []; edge_points_y_for_fitting_vis = []

    # --- Terapkan Moving Average (Smoothing) pada Theta ---
    theta_history.append(raw_theta_output)
    smoothed_theta_output = np.mean(theta_history)
    current_theta_output = smoothed_theta_output
    core_decision_for_this_frame = get_steering_decision_text(current_theta_output)

    current_decision_text = core_decision_for_this_frame
    if status_info_for_decision: current_decision_text += f" {status_info_for_decision}"
    actual_previous_theta_output = current_theta_output
    actual_previous_core_decision_text = core_decision_for_this_frame

    # ====== Visualisasi ======
    overlay = frame.copy()
    cv2.line(overlay, (center_x_frame, 0), (center_x_frame, orig_h), (255, 0, 0), 1); cv2.putText(overlay, f"Mode GPS: {current_gps_navigation_mode}", (orig_w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    y_offset = 30; info_text_size = 0.6; info_font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(overlay, f"JT (Actual): {display_jt_cm} cm", (10, y_offset), info_font, info_text_size, (0, 255, 255), 2); y_offset += 25
    cv2.putText(overlay, f"dJT: {display_djt_cm} cm/f", (10, y_offset), info_font, info_text_size, (0, 255, 255), 2); y_offset += 25
    cv2.putText(overlay, f"Target Offset: {current_target_offset_cm:.1f} cm", (10, y_offset), info_font, info_text_size, COLOR_TARGET_PATH, 2); y_offset += 25
    cv2.putText(overlay, f"Theta (Sm): {current_theta_output:.1f} deg", (10, y_offset), info_font, info_text_size, (0, 255, 255), 2); y_offset += 30

    keputusan_color = (0, 255, 0)
    if "BERHENTI" in core_decision_for_this_frame or "SANGAT MINIM" in status_info_for_decision: keputusan_color = (0, 0, 255)
    elif "HILANG" in status_info_for_decision or "Error" in status_info_for_decision or "NAVIGASI PETA" in status_info_for_decision or "Anomali" in status_info_for_decision: keputusan_color = (0, 255, 255)
    elif "Tajam" in status_info_for_decision: keputusan_color = (0, 165, 255)
    cv2.putText(overlay, f"KEPUTUSAN: {current_decision_text}", (10, y_offset), info_font, 0.7, keputusan_color, 2); y_offset += 25
    cv2.putText(overlay, f"Infer. Time: {inference_time_ms:.1f} ms", (10, y_offset), info_font, info_text_size-0.1, (200, 200, 0), 1); y_offset += 20
    loop_end_time = time.time(); time_diff_for_fps = loop_end_time - loop_start_time
    fps = 1.0 / time_diff_for_fps if time_diff_for_fps > 0 else 0
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, y_offset), info_font, info_text_size-0.1, (200, 200, 0), 1); y_offset += 20
    
    color_mask = np.zeros_like(frame); color_mask[mask_resized_to_orig == TEPI_CLASS_VALUE] = [0, 0, 255]
    
    if len(edge_points_x_for_fitting_vis) > 0:
        for i in range(len(edge_points_x_for_fitting_vis)): cv2.circle(overlay, (edge_points_x_for_fitting_vis[i], edge_points_y_for_fitting_vis[i]), RADIUS_POINTS_FOR_FIT, COLOR_POINTS_FOR_FIT, -1)
    
    y_draw_line_start_abs = int(orig_h * 0.98); y_draw_line_end_abs = int(orig_h * Y_SAMPLING_END_FACTOR)
    LINE_SEGMENT_STEP_Y = 1
    
    if found_edge_for_fitting and poly_func_for_vis is not None:
        prev_x_p_line, prev_y_p_line = -1, -1
        for y_p in range(y_draw_line_start_abs, y_draw_line_end_abs -1 , -LINE_SEGMENT_STEP_Y):
            if y_p < 0 or y_p >= orig_h : continue
            try:
                x_p = int(poly_func_for_vis(y_p))
                if 0 <= x_p < center_x_frame:
                    if prev_x_p_line != -1 : cv2.line(overlay, (prev_x_p_line, prev_y_p_line), (x_p, y_p), COLOR_FITTED_LINE, THICKNESS_FITTED_LINE)
                    prev_x_p_line = x_p; prev_y_p_line = y_p
                else: prev_x_p_line, prev_y_p_line = -1,-1
            except: prev_x_p_line, prev_y_p_line = -1,-1; pass

    if target_path_poly_for_vis is not None:
        prev_x_p_target, prev_y_p_target = -1, -1
        for y_p in range(y_draw_line_start_abs, y_draw_line_end_abs - 1, -LINE_SEGMENT_STEP_Y):
            if y_p < 0 or y_p >= orig_h: continue
            try:
                x_p = int(target_path_poly_for_vis(y_p))
                if 0 <= x_p < orig_w:
                    if prev_x_p_target != -1:
                        cv2.line(overlay, (prev_x_p_target, prev_y_p_target), (x_p, y_p), COLOR_TARGET_PATH, THICKNESS_FITTED_LINE)
                    prev_x_p_target = x_p; prev_y_p_target = y_p
                else: prev_x_p_target, prev_y_p_target = -1, -1
            except: prev_x_p_target, prev_y_p_target = -1, -1; pass

    if show_jt_reference_marker and reference_x_for_vis != -1 and vis_y_edge_for_marker != -1:
        cv2.circle(overlay, (reference_x_for_vis, vis_y_edge_for_marker), 7, (0, 255, 255), -1)
        cv2.line(overlay, (center_x_frame, vis_y_edge_for_marker), (reference_x_for_vis, vis_y_edge_for_marker), (255,255,0), 1)
    
    seg_overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
    
    # --- NEW: Logging Data dan Menulis Video ---
    current_timestamp_log = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log_data = [
        current_timestamp_log, frame_count, f"{fps:.1f}", f"{inference_time_ms:.1f}",
        f"{measured_jt_cm_log:.2f}" if measured_jt_cm_log is not None else "N/A",
        f"{djt_cm_val_log:.2f}" if djt_cm_val_log is not None else "N/A",
        f"{jt_for_fuzzy_input_log:.2f}" if jt_for_fuzzy_input_log is not None else "N/A",
        f"{raw_theta_output:.2f}", f"{current_theta_output:.2f}",
        core_decision_for_this_frame, status_info_for_decision, current_decision_text
    ]
    sheet.append(log_data)
    
    if video_writer_original is not None and video_writer_original.isOpened():
        video_writer_original.write(frame) # Tulis frame original
        
    if video_writer_overlay is not None and video_writer_overlay.isOpened():
        video_writer_overlay.write(seg_overlay) # Tulis frame dengan overlay

    cv2.imshow("Segmentasi & Navigasi Fuzzy Adaptif", seg_overlay)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('l'): current_gps_navigation_mode = "LURUS"; print("Mode GPS: LURUS")
    elif key == ord('b'): current_gps_navigation_mode = "BELOK"; print("Mode GPS: BELOK")

# --- Finalisasi ---
cap.release()
# Simpan file log Excel
workbook.save(log_filepath)
print(f"File log Excel disimpan: {log_filepath}")

# Release kedua video writer
if video_writer_overlay is not None and video_writer_overlay.isOpened():
    video_writer_overlay.release()
    print(f"Video output (overlay) disimpan: {output_video_overlay_path}")
if video_writer_original is not None and video_writer_original.isOpened():
    video_writer_original.release()
    print(f"Video output (original) disimpan: {output_video_original_path}")
    
cv2.destroyAllWindows()