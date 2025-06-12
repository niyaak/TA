import cv2
import numpy as np
import tensorflow as tf
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import time
import os 

# ====== Fuzzy Variable Definitions ======
# Perluas universe jt untuk mencakup hingga 300 cm atau lebih
jt_universe_max = 300 
jt = ctrl.Antecedent(np.arange(0, jt_universe_max + 1, 1), 'jt') # Universe dari 0 sampai 300 cm

djt = ctrl.Antecedent(np.arange(-20, 21, 1), 'djt') # djt tetap sama
theta = ctrl.Consequent(np.arange(-60, 61, 1), 'theta') # theta tetap sama

# Membership Functions for JT (Jarak Tepi) - DIUBAH TOTAL
# Dengan tj = 150-300 cm, dan asumsi normal masih di sekitar 40-90 cm

# jt['td'] (Terlalu Dekat): Tetap relatif dekat dengan 0
jt['td'] = fuzz.trapmf(jt.universe, [0,   0,  15,  25])     # 0-15 cm sangat dekat, transisi hingga 25 cm

# jt['ad'] (Agak Dekat): Menjembatani 'td' dan 'n'
jt['ad'] = fuzz.trimf(jt.universe, [20,  30,  40])       # Puncak di 30cm, rentang 20-40 cm

# jt['n'] (Normal): Sesuai diskusi sebelumnya, inti 40-90 cm, dirasakan 35-95 cm
# Kita sesuaikan sedikit agar ada ruang untuk 'aj' yang lebih jelas sebelum 'tj' baru
jt['n']  = fuzz.trapmf(jt.universe, [35,  40,  80,  90])  # NORMAL inti 40-80cm, terasa normal 35-90cm

# jt['aj'] (Agak Jauh): Menjembatani 'n' dan 'tj' baru
# 'n' berakhir di 90. 'tj' akan dimulai sekitar 140-150.
# Rentang 'aj': misalnya dari 85 cm hingga 145 cm, puncak di tengah.
jt['aj'] = fuzz.trimf(jt.universe, [85, 115, 145])      # Puncak di 115cm, rentang 85-145cm

# jt['tj'] (Terlalu Jauh): Sesuai permintaan Anda, 150-300 cm
# Menggunakan trapmf: mulai transisi sebelum 150, full 'tj' di 150-300.
jt['tj'] = fuzz.trapmf(jt.universe, [140, 150, jt_universe_max, jt_universe_max]) # Mulai dianggap jauh dari 140cm, sepenuhnya jauh dari 150cm ke atas

# --- VERIFIKASI OVERLAP (untuk pengecekan): ---
# td: [0,0,15,25]
# ad: [20,30,40]      -> overlap td-ad: 20-25
# n:  [35,40,80,90]   -> overlap ad-n: 35-40
# aj: [85,115,145]    -> overlap n-aj: 85-90
# tj: [140,150,300,300]-> overlap aj-tj: 140-145
# Semua overlap terlihat baik.

# Membership Functions for dJT & Theta (Tidak Berubah)
djt['dc'] = fuzz.trapmf(djt.universe, [-20, -20, -12, -8]); djt['dl'] = fuzz.trimf(djt.universe, [-10, -6, -2]); djt['s'] = fuzz.trimf(djt.universe, [-3, 0, 3]); djt['jl'] = fuzz.trimf(djt.universe, [2, 6, 10]); djt['jc'] = fuzz.trapmf(djt.universe, [8, 12, 20, 20])
theta['krt'] = fuzz.trapmf(theta.universe, [-60, -60, -50, -40]); theta['krs'] = fuzz.trimf(theta.universe, [-45, -30, -15]); theta['krr'] = fuzz.trimf(theta.universe, [-20, -10, -2]); theta['l'] = fuzz.trimf(theta.universe, [-5, 0, 5]); theta['knr'] = fuzz.trimf(theta.universe, [2, 10, 20]); theta['kns'] = fuzz.trimf(theta.universe, [15, 30, 45]); theta['knt'] = fuzz.trapmf(theta.universe, [40, 50, 60, 60])

# Rules (Tidak perlu diubah karena nama kategori linguistik tetap sama: td, ad, n, aj, tj)
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

MODEL_PATH = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/model/datasetbener3coba.keras'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model {MODEL_PATH} berhasil dimuat.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
input_size = (448, 256)

INPUT_SOURCE = 0 
#INPUT_SOURCE = 'D:/Documents/guekece/TUGAS AKHIR/koding/segmentasi/vidio/tes/democpt2.mp4' # Contoh jika menggunakan video

cap = cv2.VideoCapture(INPUT_SOURCE)
if not cap.isOpened():
    print(f"Error: Tidak bisa membuka sumber video/kamera: {INPUT_SOURCE}")
    exit()

# Hanya set resolusi jika menggunakan webcam
if isinstance(INPUT_SOURCE, int): 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variabel state sebelumnya
prev_jt_cm_for_djt_calc = None
actual_previous_theta_output = 0.0
actual_previous_core_decision_text = "LURUS (Inisialisasi)"
frames_edge_lost_counter = 0 

# Konstanta Kalibrasi & Logika
CM_PER_PIXEL = 0.14
MIN_ROAD_PIXELS_TO_OPERATE = 1000 
MAX_FRAMES_EDGE_LOST_BEFORE_FAR_SEARCH_OR_GPS = 3 

NORMAL_SEARCH_Y_UPPER_LIMIT_ROW_FACTOR = 0.90 
FAR_SEARCH_Y_UPPER_LIMIT_ROW_FACTOR = 0.80   

# --- Konfigurasi untuk Menyimpan Video Output ---
# Dapatkan direktori tempat skrip ini dijalankan untuk menyimpan video output di sana
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: # Terjadi jika dijalankan di lingkungan interaktif seperti Jupyter Notebook tanpa __file__
    script_dir = os.getcwd() # Gunakan direktori kerja saat ini sebagai fallback

output_video_filename = "hasil_navigasi_output.mp4" # Anda bisa ganti nama atau ekstensi (.avi)
output_video_path = os.path.join(script_dir, output_video_filename)

video_writer = None # Akan diinisialisasi setelah frame pertama dibaca
output_video_fps = 20.0  # FPS untuk video output (bisa disesuaikan)
# -----------------------------------------------

def get_steering_decision_text(theta_value):
    if theta_value < -35: return "KANAN TAJAM"
    elif theta_value < -20: return "KANAN SEDANG"
    elif theta_value < -3: return "KANAN RINGAN"
    elif theta_value <= 3: return "LURUS"
    elif theta_value <= 20: return "KIRI RINGAN"
    elif theta_value <= 35: return "KIRI SEDANG"
    else: return "KIRI TAJAM"

def find_left_edge_reference(mask, center_x, y_scan_start_row, y_scan_end_row):
    for y_row in range(y_scan_start_row, y_scan_end_row - 1, -1): 
        if y_row < 0 or y_row >= mask.shape[0]: continue
        left_half_row_segment = mask[y_row, :center_x]
        edge_class2_pixels_x_in_row = np.where(left_half_row_segment == 2)[0]
        if edge_class2_pixels_x_in_row.size > 0:
            return edge_class2_pixels_x_in_row.max(), y_row 
    return None, None

while True:
    loop_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame, stream berakhir.")
        break

    orig_h, orig_w = frame.shape[:2]
    center_x_frame = orig_w // 2

    # --- Inisialisasi VideoWriter jika ini frame pertama dan belum diinisialisasi ---
    if video_writer is None:
        frame_size_out = (orig_w, orig_h) # Ukuran frame output (lebar, tinggi)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID') # Untuk .avi
        # output_video_path_avi = os.path.join(script_dir, "hasil_navigasi_output.avi")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Untuk .mp4 (lebih umum)
        # Jika menggunakan path avi di atas, ganti output_video_path di bawah ini
        video_writer = cv2.VideoWriter(output_video_path, fourcc, output_video_fps, frame_size_out)
        if not video_writer.isOpened():
            print(f"Error: Gagal membuka VideoWriter untuk path: {output_video_path}")
            video_writer = None # Set ke None agar tidak ada error saat write/release
        else:
            print(f"Menyimpan output video ke: {output_video_path} dengan FPS: {output_video_fps}")
    # --------------------------------------------------------------------------

    current_theta_output = actual_previous_theta_output
    core_decision_for_this_frame = actual_previous_core_decision_text
    status_info_for_decision = ""       
    display_jt_cm = "N/A"
    display_djt_cm = "N/A"
    show_edge_marker = False
    reference_x_for_vis = -1 
    vis_y_edge_for_marker = -1
    inference_time_ms = 0.0
    road_percentage_display = 0.0 # Inisialisasi untuk ditampilkan jika jalan hilang
    
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
        print(f"Error saat prediksi model: {e}")
        inference_time_ms = -1.0
        status_info_for_decision = "(Prediksi Gagal)"

    y_normal_search_upper_limit = int(orig_h * NORMAL_SEARCH_Y_UPPER_LIMIT_ROW_FACTOR)
    y_far_search_upper_limit = int(orig_h * FAR_SEARCH_Y_UPPER_LIMIT_ROW_FACTOR)

    if np.sum(mask_resized_to_orig == 1) < MIN_ROAD_PIXELS_TO_OPERATE:
        current_theta_output = 0.0 
        core_decision_for_this_frame = "LURUS" 
        status_info_for_decision = "(JALAN SANGAT MINIM/HILANG!)"
        frames_edge_lost_counter = 0 
        prev_jt_cm_for_djt_calc = None
    else:
        found_reliable_edge_source = False
        reference_x_abs = -1
        y_abs_ref = -1
        
        reference_x_abs, y_abs_ref = find_left_edge_reference(mask_resized_to_orig, center_x_frame, orig_h - 1, y_normal_search_upper_limit)

        if reference_x_abs is not None:
            found_reliable_edge_source = True
            frames_edge_lost_counter = 0
        else:
            frames_edge_lost_counter += 1
            if frames_edge_lost_counter > MAX_FRAMES_EDGE_LOST_BEFORE_FAR_SEARCH_OR_GPS:
                reference_x_abs, y_abs_ref = find_left_edge_reference(mask_resized_to_orig, center_x_frame, y_normal_search_upper_limit - 1, y_far_search_upper_limit)
                if reference_x_abs is not None:
                    found_reliable_edge_source = True
                    frames_edge_lost_counter = 0 
                    status_info_for_decision = "(Ref: Tepi Jauh)" 
        
        if found_reliable_edge_source:
            reference_x_for_vis = reference_x_abs
            vis_y_edge_for_marker = y_abs_ref
            jt_pixel_val = center_x_frame - reference_x_abs
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
                if not status_info_for_decision : 
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
            if frames_edge_lost_counter <= MAX_FRAMES_EDGE_LOST_BEFORE_FAR_SEARCH_OR_GPS:
                status_info_for_decision = f"(TEPI HILANG SEMENTARA ({frames_edge_lost_counter}))"
            else: 
                current_theta_output = 0.0 
                core_decision_for_this_frame = "LURUS" 
                status_info_for_decision = "(NAVIGASI PETA AKTIF)"
            prev_jt_cm_for_djt_calc = None 
            display_jt_cm = "N/A"
            display_djt_cm = "N/A (Tepi Hilang)"
            show_edge_marker = False
            reference_x_for_vis = -1
            
    current_decision_text = core_decision_for_this_frame
    if status_info_for_decision:
        current_decision_text += f" {status_info_for_decision}"

    actual_previous_theta_output = current_theta_output
    actual_previous_core_decision_text = core_decision_for_this_frame

    # ====== Visualisasi ======
    overlay = frame.copy()
    cv2.line(overlay, (center_x_frame, 0), (center_x_frame, orig_h), (255, 0, 0), 1)
    cv2.line(overlay, (0, y_normal_search_upper_limit), (center_x_frame, y_normal_search_upper_limit), (0, 255, 125), 1)
    cv2.putText(overlay, "Batas Prio 1 (Dekat)", (5, y_normal_search_upper_limit - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 125),1)
    if y_far_search_upper_limit < y_normal_search_upper_limit :
        cv2.line(overlay, (0, y_far_search_upper_limit), (center_x_frame, y_far_search_upper_limit), (255, 125, 0), 1)
        cv2.putText(overlay, "Batas Prio 2 (Jauh)", (5, y_far_search_upper_limit - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 125, 0),1)

    y_offset = 30; info_text_size = 0.6; info_font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, f"JT: {display_jt_cm} cm", (10, y_offset), info_font, info_text_size, (0, 255, 255), 2); y_offset += 25
    cv2.putText(overlay, f"dJT: {display_djt_cm} cm/f", (10, y_offset), info_font, info_text_size, (0, 255, 255), 2); y_offset += 25
    cv2.putText(overlay, f"Theta: {current_theta_output:.1f} deg", (10, y_offset), info_font, info_text_size, (0, 255, 255), 2); y_offset += 30
    
    keputusan_color = (0, 255, 0) 
    if "BERHENTI" in core_decision_for_this_frame or "SANGAT MINIM" in status_info_for_decision: 
        keputusan_color = (0, 0, 255) 
    elif "HILANG" in status_info_for_decision or "Error" in status_info_for_decision or "NAVIGASI PETA" in status_info_for_decision: 
        keputusan_color = (0, 255, 255) 
    cv2.putText(overlay, f"KEPUTUSAN: {current_decision_text}", (10, y_offset), info_font, 0.7, keputusan_color, 2); y_offset += 25
    cv2.putText(overlay, f"Infer. Time: {inference_time_ms:.1f} ms", (10, y_offset), info_font, info_text_size-0.1, (200, 200, 0), 1); y_offset += 20
    
    loop_end_time = time.time()
    time_diff_for_fps = loop_end_time - loop_start_time
    fps = 1.0 / time_diff_for_fps if time_diff_for_fps > 0 else 0
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, y_offset), info_font, info_text_size-0.1, (200, 200, 0), 1); y_offset += 20

    color_mask = np.zeros_like(frame); color_mask[mask_resized_to_orig == 1] = [0, 255, 0]; color_mask[mask_resized_to_orig == 2] = [0, 0, 255]

    if show_edge_marker and reference_x_for_vis != -1 and vis_y_edge_for_marker != -1:
        cv2.circle(overlay, (reference_x_for_vis, vis_y_edge_for_marker), 7, (0, 255, 255), -1)
        cv2.line(overlay, (center_x_frame, vis_y_edge_for_marker), (reference_x_for_vis, vis_y_edge_for_marker), (255,255,0), 1)

    seg_overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
    
    # --- Tulis frame ke video output ---
    if video_writer is not None and video_writer.isOpened():
        video_writer.write(seg_overlay)
    # ------------------------------------

    cv2.imshow("Segmentasi & Navigasi Fuzzy", seg_overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Setelah loop selesai ---
cap.release()
if video_writer is not None and video_writer.isOpened(): # Pastikan ada dan terbuka sebelum release
    video_writer.release()
    print(f"Video output disimpan sebagai: {output_video_path}")
cv2.destroyAllWindows()