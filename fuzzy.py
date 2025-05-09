import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

# Definisi variabel input dan output
jt = np.arange(0, 121, 1)    # Jarak ke Tepi Kiri (cm)
djt = np.arange(-20, 21, 1)  # Perubahan Jarak ke Tepi Kiri (cm/frame)
theta = np.arange(-60, 61, 1)  # Arah Kendaraan (derajat)

# Membership Function untuk Jarak ke Tepi Kiri (JT)
jt_td = fuzz.trapmf(jt, [0, 0, 10, 25])
jt_ad = fuzz.trimf(jt, [15, 30, 45])
jt_n = fuzz.trimf(jt, [35, 55, 75])
jt_aj = fuzz.trimf(jt, [60, 80, 100])
jt_tj = fuzz.trapmf(jt, [90, 110, 120, 120])

# Membership Function untuk Perubahan Jarak ke Tepi Kiri (ΔJT)
djt_dc = fuzz.trapmf(djt, [-20, -20, -12, -8])
djt_dl = fuzz.trimf(djt, [-10, -6, -2])
djt_s = fuzz.trimf(djt, [-3, 0, 3])
djt_jl = fuzz.trimf(djt, [2, 6, 10])
djt_jc = fuzz.trapmf(djt, [8, 12, 20, 20])

# Membership Function untuk Arah Kendaraan (ΔΘ) - Diperbaiki
theta_krt = fuzz.trapmf(theta, [-60, -60, -50, -40])  # Belok Kiri Tajam
theta_krs = fuzz.trimf(theta, [-45, -30, -15])  # Belok Kiri Sedang
theta_krr = fuzz.trimf(theta, [-20, -10, -2])  # Belok Kiri Ringan
theta_l = fuzz.trimf(theta, [-5, 0, 5])  # Lurus
theta_knr = fuzz.trimf(theta, [2, 10, 20])  # Belok Kanan Ringan
theta_kns = fuzz.trimf(theta, [15, 30, 45])  # Belok Kanan Sedang
theta_knt = fuzz.trapmf(theta, [40, 50, 60, 60])  # Belok Kanan Tajam

# Fungsi untuk menampilkan grafik
def plot_membership1(x, mf, title, labels):
    plt.figure(figsize=(8, 4))
    for i, (mf_func, label) in enumerate(zip(mf, labels)):
        plt.plot(x, mf_func, label=label)
    plt.title(title)
    plt.xlabel("Jarak (cm)")
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()
    plt.grid()
    plt.show()

# Fungsi untuk menampilkan grafik
def plot_membership2(x, mf, title, labels):
    plt.figure(figsize=(8, 4))
    for i, (mf_func, label) in enumerate(zip(mf, labels)):
        plt.plot(x, mf_func, label=label)
    plt.title(title)
    plt.xlabel("Jarak (cm/frame)")
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()
    plt.grid()
    plt.show()

# Fungsi untuk menampilkan grafik
def plot_membership3(x, mf, title, labels):
    plt.figure(figsize=(8, 4))
    for i, (mf_func, label) in enumerate(zip(mf, labels)):
        plt.plot(x, mf_func, label=label)
    plt.title(title)
    plt.xlabel("Derajat (°)")
    plt.ylabel("Derajat Keanggotaan")
    plt.legend()
    plt.grid()
    plt.show()

# Plot Membership Function
plot_membership1(jt, [jt_td, jt_ad, jt_n, jt_aj, jt_tj], "Membership Function - Jarak ke Tepi Kiri (JT)", 
                ["Terlalu Dekat (TD)", "Agak Dekat (AD)", "Normal (N)", "Agak Jauh (AJ)", "Terlalu Jauh (TJ)"])

plot_membership2(djt, [djt_dc, djt_dl, djt_s, djt_jl, djt_jc], "Membership Function - Perubahan Jarak ke Tepi Kiri (PJT)", 
                ["Mendekat Cepat (DC)", "Mendekat Lambat (DL)", "Stabil (S)", "Menjauh Lambat (JL)", "Menjauh Cepat (JC)"])

# Plot Membership Function - Arah Kendaraan
plot_membership3(theta, [theta_krt, theta_krs, theta_krr, theta_l, theta_knr, theta_kns, theta_knt], 
                "Membership Function - Arah Kendaraan (AK)", 
                ["Belok Kiri Tajam (KRT)", "Belok Kiri Sedang (KRS)", "Belok Kiri Ringan (KRR)", 
                 "Lurus (L)", "Belok Kanan Ringan (KNR)", "Belok Kanan Sedang (KNS)", "Belok Kanan Tajam (KNT)"])
