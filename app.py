import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import json
import pickle
from datetime import datetime, date
from PIL import Image
import io
import base64

# ============================================================
# KONFIGURASI
# ============================================================
st.set_page_config(
    page_title="Sistem Absensi Wajah",
    page_icon="📷",
    layout="wide"
)

DATA_DIR = "data"
FOTO_DIR = os.path.join(DATA_DIR, "foto")
MODEL_FILE = os.path.join(DATA_DIR, "model_lbph.pkl")
LABEL_FILE = os.path.join(DATA_DIR, "labels.json")
ABSENSI_FILE = os.path.join(DATA_DIR, "absensi.csv")
MAHASISWA_FILE = os.path.join(DATA_DIR, "mahasiswa.csv")

for d in [DATA_DIR, FOTO_DIR]:
    os.makedirs(d, exist_ok=True)

ADMIN_USER = "admin"
ADMIN_PASS = "admin123"

THRESHOLD = 60  # Makin kecil makin ketat (0-100). 60 = seimbang akurasi & toleransi

# ============================================================
# FUNGSI UTILITAS
# ============================================================

def load_labels():
    if os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, "r") as f:
            return json.load(f)
    return {}

def save_labels(labels):
    with open(LABEL_FILE, "w") as f:
        json.dump(labels, f, ensure_ascii=False)

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

def load_mahasiswa():
    if os.path.exists(MAHASISWA_FILE):
        return pd.read_csv(MAHASISWA_FILE)
    return pd.DataFrame(columns=["nim", "nama", "kelas"])

def save_mahasiswa(df):
    df.to_csv(MAHASISWA_FILE, index=False)

def load_absensi():
    if os.path.exists(ABSENSI_FILE):
        try:
            df = pd.read_csv(ABSENSI_FILE)
            return df
        except:
            pass
    return pd.DataFrame(columns=["tanggal", "waktu", "nim", "nama", "kelas", "status"])

def save_absensi(df):
    df.to_csv(ABSENSI_FILE, index=False)

def detect_face(image_array):
    """Deteksi wajah menggunakan Haar Cascade."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    return gray, faces

def preprocess_face(gray, x, y, w, h, size=(150, 150)):
    """Crop dan resize wajah."""
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, size)
    face = cv2.equalizeHist(face)  # normalisasi pencahayaan
    return face

def train_model():
    """Latih model LBPH dari semua foto yang tersimpan."""
    labels = load_labels()
    if not labels:
        return None

    faces = []
    ids = []
    label_to_id = {nim: idx for idx, nim in enumerate(labels.keys())}

    for nim, info in labels.items():
        nim_dir = os.path.join(FOTO_DIR, nim)
        if not os.path.exists(nim_dir):
            continue
        for fname in os.listdir(nim_dir):
            if fname.endswith(".jpg"):
                img_path = os.path.join(nim_dir, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (150, 150))
                    img = cv2.equalizeHist(img)
                    faces.append(img)
                    ids.append(label_to_id[nim])

    if len(faces) < 1:
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=THRESHOLD
    )
    recognizer.train(faces, np.array(ids))
    save_model(recognizer)
    return recognizer

def recognize_face(gray_face):
    """Kenali wajah, return (nim, confidence) atau (None, None)."""
    model = load_model()
    labels = load_labels()
    if model is None or not labels:
        return None, None

    id_to_nim = {idx: nim for idx, nim in enumerate(labels.keys())}
    try:
        label_id, confidence = model.predict(gray_face)
        if confidence <= THRESHOLD:
            nim = id_to_nim.get(label_id)
            return nim, confidence
        return None, confidence
    except:
        return None, None

# ============================================================
# SESSION STATE
# ============================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None

# ============================================================
# HALAMAN LOGIN
# ============================================================
def halaman_login():
    st.title("🎓 Sistem Absensi Mahasiswa")
    st.subheader("Silakan login terlebih dahulu")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        role = st.radio("Login sebagai:", ["Admin", "Mahasiswa"], horizontal=True)
        username = st.text_input("Username / NIM")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if role == "Admin":
                if username == ADMIN_USER and password == ADMIN_PASS:
                    st.session_state.logged_in = True
                    st.session_state.role = "admin"
                    st.rerun()
                else:
                    st.error("Username atau password salah!")
            else:
                mhs = load_mahasiswa()
                match = mhs[mhs["nim"] == username]
                if not match.empty and password == username:  # password = NIM
                    st.session_state.logged_in = True
                    st.session_state.role = "mahasiswa"
                    st.session_state.nim = username
                    st.session_state.nama = match.iloc[0]["nama"]
                    st.rerun()
                else:
                    st.error("NIM tidak terdaftar atau password salah! (Password = NIM)")

# ============================================================
# HALAMAN DAFTAR MAHASISWA (ADMIN)
# ============================================================
def halaman_daftar_mahasiswa():
    st.header("👥 Daftar Mahasiswa")

    tab1, tab2 = st.tabs(["➕ Tambah Mahasiswa", "📋 Data Mahasiswa"])

    with tab1:
        with st.form("form_daftar"):
            nim = st.text_input("NIM")
            nama = st.text_input("Nama Lengkap")
            kelas = st.text_input("Kelas")
            foto_files = st.file_uploader(
                "Upload Foto Wajah (3-5 foto dari sudut berbeda)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )
            submitted = st.form_submit_button("Daftarkan")

        if submitted:
            if not nim or not nama or not kelas:
                st.error("Semua field wajib diisi!")
            elif len(foto_files) < 1:
                st.error("Upload minimal 1 foto!")
            else:
                mhs = load_mahasiswa()
                if nim in mhs["nim"].values:
                    st.warning(f"NIM {nim} sudah terdaftar. Data akan diupdate.")
                    mhs = mhs[mhs["nim"] != nim]

                nim_dir = os.path.join(FOTO_DIR, nim)
                os.makedirs(nim_dir, exist_ok=True)

                saved = 0
                for i, foto in enumerate(foto_files):
                    img = Image.open(foto).convert("RGB")
                    img_array = np.array(img)
                    gray, faces = detect_face(img_array)

                    if len(faces) == 0:
                        st.warning(f"Foto {i+1}: Wajah tidak terdeteksi, dilewati.")
                        continue

                    x, y, w, h = faces[0]
                    face_img = preprocess_face(gray, x, y, w, h)
                    cv2.imwrite(os.path.join(nim_dir, f"foto_{i}.jpg"), face_img)
                    saved += 1

                if saved == 0:
                    st.error("Tidak ada wajah yang berhasil disimpan dari semua foto!")
                else:
                    labels = load_labels()
                    labels[nim] = {"nama": nama, "kelas": kelas}
                    save_labels(labels)

                    new_row = pd.DataFrame([{"nim": nim, "nama": nama, "kelas": kelas}])
                    mhs = pd.concat([mhs, new_row], ignore_index=True)
                    save_mahasiswa(mhs)

                    with st.spinner("Melatih model pengenalan wajah..."):
                        model = train_model()

                    if model:
                        st.success(f"✅ {nama} berhasil didaftarkan! {saved} foto wajah tersimpan.")
                    else:
                        st.warning("Mahasiswa terdaftar tapi model belum bisa dilatih.")

    with tab2:
        mhs = load_mahasiswa()
        if mhs.empty:
            st.info("Belum ada mahasiswa terdaftar.")
        else:
            st.dataframe(mhs, use_container_width=True)
            st.caption(f"Total: {len(mhs)} mahasiswa")

            # Hapus mahasiswa
            with st.expander("🗑️ Hapus Mahasiswa"):
                nim_hapus = st.selectbox("Pilih NIM", mhs["nim"].tolist())
                if st.button("Hapus", type="primary"):
                    mhs = mhs[mhs["nim"] != nim_hapus]
                    save_mahasiswa(mhs)
                    labels = load_labels()
                    labels.pop(nim_hapus, None)
                    save_labels(labels)
                    nim_dir = os.path.join(FOTO_DIR, nim_hapus)
                    if os.path.exists(nim_dir):
                        import shutil
                        shutil.rmtree(nim_dir)
                    train_model()
                    st.success(f"Mahasiswa {nim_hapus} berhasil dihapus!")
                    st.rerun()

# ============================================================
# HALAMAN ABSENSI (KAMERA)
# ============================================================
def halaman_absensi():
    st.header("📷 Absensi dengan Kamera")

    model = load_model()
    labels = load_labels()

    if model is None or not labels:
        st.warning("⚠️ Belum ada model/mahasiswa terdaftar. Daftarkan mahasiswa dulu!")
        return

    st.info("📌 Ambil foto menggunakan kamera di bawah. Pastikan wajah terlihat jelas dan pencahayaan cukup.")

    img_file = st.camera_input("Ambil Foto untuk Absensi")

    if img_file:
        img = Image.open(img_file).convert("RGB")
        img_array = np.array(img)
        gray, faces = detect_face(img_array)

        if len(faces) == 0:
            st.error("❌ Wajah tidak terdeteksi. Pastikan wajah terlihat jelas!")
            return

        # Gambar kotak wajah
        img_display = img_array.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_display, (x, y), (x+w, y+h), (0, 255, 0), 2)

        x, y, w, h = faces[0]
        face_gray = preprocess_face(gray, x, y, w, h)

        nim, confidence = recognize_face(face_gray)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_display, caption="Deteksi Wajah", use_container_width=True)

        with col2:
            if nim and nim in labels:
                info = labels[nim]
                st.success(f"✅ **Wajah Dikenali!**")
                st.metric("Nama", info["nama"])
                st.metric("NIM", nim)
                st.metric("Kelas", info["kelas"])
                st.metric("Kepercayaan", f"{100 - confidence:.1f}%")

                # Cek sudah absen hari ini
                absensi = load_absensi()
                today = date.today().strftime("%Y-%m-%d")
                sudah = absensi[(absensi["nim"] == nim) & (absensi["tanggal"] == today)]

                if not sudah.empty:
                    st.warning(f"⚠️ {info['nama']} sudah absen hari ini pukul {sudah.iloc[-1]['waktu']}")
                else:
                    if st.button("✅ Konfirmasi Absensi", type="primary", use_container_width=True):
                        now = datetime.now()
                        new_row = pd.DataFrame([{
                            "tanggal": today,
                            "waktu": now.strftime("%H:%M:%S"),
                            "nim": nim,
                            "nama": info["nama"],
                            "kelas": info["kelas"],
                            "status": "Hadir"
                        }])
                        absensi = pd.concat([absensi, new_row], ignore_index=True)
                        save_absensi(absensi)
                        st.success(f"🎉 Absensi {info['nama']} berhasil dicatat!")
                        st.balloons()
            else:
                st.error(f"❌ **Wajah Tidak Dikenali**")
                st.write(f"Confidence: {confidence:.1f} (threshold: {THRESHOLD})")
                st.write("Kemungkinan penyebab:")
                st.write("- Bukan mahasiswa terdaftar")
                st.write("- Pencahayaan kurang")
                st.write("- Sudut wajah berbeda")

# ============================================================
# HALAMAN REKAP ABSENSI
# ============================================================
def halaman_rekap():
    st.header("📊 Rekap Absensi")

    absensi = load_absensi()

    if absensi.empty:
        st.info("Belum ada data absensi.")
        return

    tab1, tab2, tab3 = st.tabs(["📋 Semua Data", "📅 Filter Tanggal", "🗑️ Hapus Data"])

    with tab1:
        st.dataframe(absensi, use_container_width=True)
        st.caption(f"Total: {len(absensi)} record absensi")

        csv = absensi.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "rekap_absensi.csv", "text/csv")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            tgl_mulai = st.date_input("Dari Tanggal")
        with col2:
            tgl_akhir = st.date_input("Sampai Tanggal")

        if st.button("Filter"):
            mask = (absensi["tanggal"] >= str(tgl_mulai)) & (absensi["tanggal"] <= str(tgl_akhir))
            filtered = absensi[mask]
            if filtered.empty:
                st.info("Tidak ada data pada rentang tanggal tersebut.")
            else:
                st.dataframe(filtered, use_container_width=True)
                csv2 = filtered.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Filtered CSV", csv2, "rekap_filtered.csv", "text/csv")

    with tab3:
        st.warning("⚠️ Tindakan ini tidak bisa dibatalkan!")
        konfirmasi = st.text_input("Ketik HAPUS SEMUA untuk konfirmasi")
        if st.button("Hapus Semua Data Absensi", type="primary"):
            if konfirmasi == "HAPUS SEMUA":
                empty_df = pd.DataFrame(columns=["tanggal", "waktu", "nim", "nama", "kelas", "status"])
                save_absensi(empty_df)
                st.success("Semua data absensi telah dihapus!")
                st.rerun()
            else:
                st.error("Konfirmasi salah! Ketik HAPUS SEMUA dengan benar.")

# ============================================================
# HALAMAN ABSENSI MAHASISWA (SELF)
# ============================================================
def halaman_absensi_mahasiswa():
    st.header(f"👤 Absensi Mandiri - {st.session_state.nama}")

    nim = st.session_state.nim
    labels = load_labels()
    model = load_model()

    if model is None or nim not in labels:
        st.warning("Data wajah Anda belum terdaftar. Hubungi admin!")
        return

    img_file = st.camera_input("Ambil Foto untuk Absensi")

    if img_file:
        img = Image.open(img_file).convert("RGB")
        img_array = np.array(img)
        gray, faces = detect_face(img_array)

        if len(faces) == 0:
            st.error("❌ Wajah tidak terdeteksi!")
            return

        x, y, w, h = faces[0]
        face_gray = preprocess_face(gray, x, y, w, h)
        recognized_nim, confidence = recognize_face(face_gray)

        if recognized_nim == nim:
            info = labels[nim]
            absensi = load_absensi()
            today = date.today().strftime("%Y-%m-%d")
            sudah = absensi[(absensi["nim"] == nim) & (absensi["tanggal"] == today)]

            if not sudah.empty:
                st.warning(f"Anda sudah absen hari ini pukul {sudah.iloc[-1]['waktu']}")
            else:
                st.success(f"✅ Wajah dikenali sebagai {info['nama']}")
                if st.button("Konfirmasi Absensi", type="primary"):
                    now = datetime.now()
                    new_row = pd.DataFrame([{
                        "tanggal": today,
                        "waktu": now.strftime("%H:%M:%S"),
                        "nim": nim,
                        "nama": info["nama"],
                        "kelas": info["kelas"],
                        "status": "Hadir"
                    }])
                    absensi = pd.concat([absensi, new_row], ignore_index=True)
                    save_absensi(absensi)
                    st.success("🎉 Absensi berhasil!")
                    st.balloons()
        else:
            st.error("❌ Wajah tidak cocok dengan akun Anda!")

# ============================================================
# MAIN APP
# ============================================================
def main():
    if not st.session_state.logged_in:
        halaman_login()
        return

    with st.sidebar:
        st.title("📷 Sistem Absensi")
        st.write(f"**Role:** {st.session_state.role.title()}")

        if st.session_state.role == "admin":
            menu = st.radio("Menu", [
                "📷 Absensi",
                "👥 Daftar Mahasiswa",
                "📊 Rekap Absensi"
            ])
        else:
            menu = st.radio("Menu", ["📷 Absensi Mandiri"])

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.rerun()

    if st.session_state.role == "admin":
        if menu == "📷 Absensi":
            halaman_absensi()
        elif menu == "👥 Daftar Mahasiswa":
            halaman_daftar_mahasiswa()
        elif menu == "📊 Rekap Absensi":
            halaman_rekap()
    else:
        halaman_absensi_mahasiswa()

if __name__ == "__main__":
    main()
