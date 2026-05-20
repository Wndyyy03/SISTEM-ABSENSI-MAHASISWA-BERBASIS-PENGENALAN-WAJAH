"""
PENDAFTARAN WAJAH MAHASISWA - Versi Stabil Windows
"""

import cv2
import os
import pickle
import numpy as np

DATA_DIR = "data_wajah"
DATABASE_FILE = "database_wajah.pkl"
os.makedirs(DATA_DIR, exist_ok=True)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def ambil_fitur_wajah(gray_roi):
    roi = cv2.resize(gray_roi, (128, 128))
    roi = cv2.equalizeHist(roi)
    return roi.flatten().astype(np.float64)

def buka_kamera():
    """Coba berbagai cara buka kamera sampai berhasil."""
    # Coba index 0 dan 1 dengan berbagai backend
    for idx in [0, 1]:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Kamera berhasil dibuka (index={idx})")
                    return cap
                cap.release()
    return None

def daftar_wajah():
    nama = input("\nNama Lengkap : ").strip()
    nim  = input("NIM          : ").strip()

    if not nama or not nim:
        print("Nama dan NIM tidak boleh kosong!")
        return

    print("\nMembuka kamera, harap tunggu...")
    
    cap = buka_kamera()
    if cap is None:
        print("GAGAL membuka kamera!")
        print("Pastikan:")
        print("  1. Webcam terhubung ke laptop")
        print("  2. Tidak ada aplikasi lain yang pakai kamera (Zoom, Teams, dll)")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    print(f"Kamera aktif! Hadapkan wajah ke kamera.")
    print("Tekan SPASI untuk foto | Q untuk keluar")

    encoding_wajah = None
    foto_diambil   = False
    gagal_baca     = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            gagal_baca += 1
            if gagal_baca > 30:
                print("Kamera bermasalah, keluar otomatis.")
                break
            continue
        gagal_baca = 0

        tampilan = frame.copy()
        h, w = tampilan.shape[:2]

        # Deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        wajah_cv = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
        )

        for (x, y, fw, fh) in wajah_cv:
            cv2.rectangle(tampilan, (x,y), (x+fw,y+fh), (0,255,0), 2)

        # UI
        cv2.rectangle(tampilan, (0,0), (w,65), (20,20,20), -1)
        cv2.putText(tampilan, f"{nama} | {nim}",
                    (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        if len(wajah_cv) > 0:
            cv2.putText(tampilan, "Wajah terdeteksi! Tekan SPASI",
                        (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        else:
            cv2.putText(tampilan, "Hadapkan wajah ke kamera...",
                        (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

        cv2.rectangle(tampilan, (0,h-30), (w,h), (20,20,20), -1)
        cv2.putText(tampilan, "SPASI = Foto   |   Q = Keluar",
                    (10,h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        cv2.imshow("Pendaftaran Wajah - Tekan SPASI untuk foto", tampilan)

        key = cv2.waitKey(30) & 0xFF  # 30ms delay, lebih stabil

        if key in [ord('q'), ord('Q'), 27]:
            print("Dibatalkan.")
            break

        elif key == ord(' '):
            print("Mengambil foto...")

            # Baca beberapa frame agar stabil
            for _ in range(5):
                ret2, frame2 = cap.read()

            if not ret2 or frame2 is None:
                print("Gagal ambil foto, coba lagi.")
                continue

            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            wajah2 = FACE_CASCADE.detectMultiScale(
                gray2, scaleFactor=1.2, minNeighbors=5, minSize=(80,80)
            )

            if len(wajah2) == 0:
                print("Wajah tidak terdeteksi! Pastikan cahaya cukup.")
                continue

            if len(wajah2) > 1:
                print("Lebih dari 1 wajah! Pastikan hanya 1 orang.")
                continue

            (x, y, fw, fh) = wajah2[0]
            roi_gray = gray2[y:y+fh, x:x+fw]
            encoding_wajah = ambil_fitur_wajah(roi_gray)

            nama_file = f"{nim}_{nama.replace(' ','_')}.jpg"
            cv2.imwrite(os.path.join(DATA_DIR, nama_file), frame2)

            print(f"Foto tersimpan!")
            foto_diambil = True
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Flush event kamera

    if not foto_diambil or encoding_wajah is None:
        return

    database = {}
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, "rb") as f:
            database = pickle.load(f)

    database[nim] = {"nama": nama, "nim": nim, "encoding": encoding_wajah}

    with open(DATABASE_FILE, "wb") as f:
        pickle.dump(database, f)

    print(f"\nBERHASIL! {nama} ({nim}) terdaftar!")
    print(f"Total: {len(database)} mahasiswa\n")


def lihat_daftar():
    if not os.path.exists(DATABASE_FILE):
        print("\nBelum ada mahasiswa terdaftar.\n")
        return
    with open(DATABASE_FILE, "rb") as f:
        database = pickle.load(f)
    print(f"\n=== DAFTAR MAHASISWA ({len(database)} orang) ===")
    for i, (nim, data) in enumerate(database.items(), 1):
        print(f"{i}. {nim} - {data['nama']}")
    print()


if __name__ == "__main__":
    while True:
        print("\n=== MENU PENDAFTARAN WAJAH ===")
        print("1. Daftarkan mahasiswa baru")
        print("2. Lihat daftar mahasiswa")
        print("3. Keluar")
        pilih = input("Pilihan (1/2/3): ").strip()
        if pilih == "1":
            daftar_wajah()
        elif pilih == "2":
            lihat_daftar()
        elif pilih == "3":
            break
        else:
            print("Pilihan tidak valid.")
