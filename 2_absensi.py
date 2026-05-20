"""
SISTEM ABSENSI PENGENALAN WAJAH
Menggunakan OpenCV + MediaPipe (tanpa face_recognition)
"""

import cv2
import os
import pickle
import numpy as np
import mediapipe as mp
import csv
from datetime import datetime

DATABASE_FILE = "database_wajah.pkl"
REKAP_FILE    = "rekap_absensi.csv"

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Landmark mata
MATA_KIRI  = [362, 385, 387, 263, 373, 380]
MATA_KANAN = [33,  160, 158, 133, 153, 144]
KEDIP_THRESHOLD = 0.22
KEDIP_REQUIRED  = 2


def hitung_ear(landmarks, indeks, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indeks]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C != 0 else 0


def ambil_fitur_wajah(gray_roi):
    roi = cv2.resize(gray_roi, (128, 128))
    roi = cv2.equalizeHist(roi)
    return roi.flatten().astype(np.float64)


def cocokkan_wajah(encoding_baru, database, threshold=0.35):
    """Cocokkan encoding wajah dengan database pakai korelasi."""
    best_nim  = None
    best_nama = None
    best_skor = -1

    for nim, data in database.items():
        enc_db = data["encoding"]
        # Normalisasi
        a = encoding_baru / (np.linalg.norm(encoding_baru) + 1e-6)
        b = enc_db / (np.linalg.norm(enc_db) + 1e-6)
        skor = np.dot(a, b)

        if skor > best_skor:
            best_skor = skor
            best_nim  = nim
            best_nama = data["nama"]

    if best_skor >= threshold:
        return best_nim, best_nama, best_skor
    return None, None, best_skor


def catat_absensi(nim, nama):
    sekarang = datetime.now()
    tanggal  = sekarang.strftime("%Y-%m-%d")
    waktu    = sekarang.strftime("%H:%M:%S")

    if os.path.exists(REKAP_FILE):
        with open(REKAP_FILE, "r") as f:
            for row in csv.reader(f):
                if len(row) >= 3 and row[0] == nim and row[2] == tanggal:
                    return False, "sudah_absen"

    file_baru = not os.path.exists(REKAP_FILE)
    with open(REKAP_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if file_baru:
            writer.writerow(["NIM","Nama","Tanggal","Waktu","Status"])
        writer.writerow([nim, nama, tanggal, waktu, "HADIR"])
    return True, waktu


def jalankan_absensi():
    if not os.path.exists(DATABASE_FILE):
        print("Database belum ada! Jalankan dulu: python 1_daftar_wajah.py")
        return

    with open(DATABASE_FILE, "rb") as f:
        database = pickle.load(f)

    print(f"Database dimuat: {len(database)} mahasiswa.")
    print("Kamera aktif. Hadapkan wajah & kedipkan mata 2x.\nTekan Q untuk keluar.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    jumlah_kedip   = 0
    sedang_kedip   = False
    wajah_dikenali = None
    fase           = "SCAN"
    pesan          = "Arahkan wajah ke kamera..."
    warna          = (200, 200, 200)
    timer_reset    = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame    = cv2.flip(frame, 1)
        tampilan = frame.copy()
        h, w     = frame.shape[:2]
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # FASE SCAN
        if fase == "SCAN":
            wajah_cv = FACE_CASCADE.detectMultiScale(
                gray, 1.1, 5, minSize=(80,80)
            )
            if len(wajah_cv) > 0:
                (x, y, fw, fh) = wajah_cv[0]
                roi = gray[y:y+fh, x:x+fw]
                enc = ambil_fitur_wajah(roi)
                nim, nama, skor = cocokkan_wajah(enc, database)

                cv2.rectangle(tampilan, (x,y), (x+fw,y+fh),
                              (0,255,0) if nim else (0,0,255), 2)

                if nim:
                    wajah_dikenali = {"nim": nim, "nama": nama}
                    fase  = "KEDIP"
                    jumlah_kedip = 0
                    pesan = f"Halo {nama}! Kedipkan mata {KEDIP_REQUIRED}x"
                    warna = (0, 255, 255)
                else:
                    pesan = f"Wajah tidak dikenali (skor:{skor:.2f})"
                    warna = (0, 0, 255)
            else:
                pesan = "Arahkan wajah ke kamera..."
                warna = (200, 200, 200)

        # FASE KEDIP (anti-spoofing)
        elif fase == "KEDIP":
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hasil = face_mesh.process(rgb_frame)

            if hasil.multi_face_landmarks:
                lm = hasil.multi_face_landmarks[0].landmark
                ear_k = hitung_ear(lm, MATA_KIRI,  w, h)
                ear_ka = hitung_ear(lm, MATA_KANAN, w, h)
                ear   = (ear_k + ear_ka) / 2.0

                if ear < KEDIP_THRESHOLD:
                    sedang_kedip = True
                elif sedang_kedip:
                    jumlah_kedip += 1
                    sedang_kedip  = False

                sisa  = KEDIP_REQUIRED - jumlah_kedip
                pesan = f"Kedipkan mata {sisa}x lagi..."
                warna = (0, 255, 255)

                for i in range(KEDIP_REQUIRED):
                    warna_dot = (0,255,0) if i < jumlah_kedip else (100,100,100)
                    cv2.circle(tampilan, (w//2 - 20 + i*40, h-40), 12, warna_dot, -1)

                if jumlah_kedip >= KEDIP_REQUIRED:
                    fase = "PROSES"
            else:
                pesan = "Wajah hilang, ulangi..."
                warna = (0, 165, 255)
                fase  = "SCAN"
                wajah_dikenali = None

        # FASE PROSES
        elif fase == "PROSES":
            berhasil, info = catat_absensi(
                wajah_dikenali["nim"], wajah_dikenali["nama"]
            )
            if berhasil:
                pesan = f"TERCATAT! {wajah_dikenali['nama']} - {info}"
                warna = (0, 255, 0)
                print(f"Absensi: {wajah_dikenali['nama']} ({wajah_dikenali['nim']}) - {info}")
            else:
                pesan = f"{wajah_dikenali['nama']} sudah absen hari ini!"
                warna = (0, 165, 255)
            fase = "SUKSES"
            timer_reset = 90

        # FASE SUKSES
        elif fase == "SUKSES":
            timer_reset -= 1
            if timer_reset <= 0:
                fase = "SCAN"
                wajah_dikenali = None
                jumlah_kedip   = 0
                pesan = "Siap untuk mahasiswa berikutnya..."
                warna = (200, 200, 200)

        # UI
        cv2.rectangle(tampilan, (0,0), (w,70), (20,20,20), -1)
        cv2.putText(tampilan, "SISTEM ABSENSI - UNISMUH MAKASSAR",
                    (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(tampilan, datetime.now().strftime("%d %B %Y  |  %H:%M:%S"),
                    (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1)

        cv2.rectangle(tampilan, (0,h-60), (w,h), (20,20,20), -1)
        cv2.putText(tampilan, pesan,
                    (10,h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, warna, 2)
        cv2.putText(tampilan, "Q = Keluar",
                    (10,h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,120), 1)

        cv2.imshow("Absensi Pengenalan Wajah", tampilan)

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nData tersimpan di: {REKAP_FILE}")


if __name__ == "__main__":
    jalankan_absensi()
