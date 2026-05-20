"""
SISTEM ABSENSI MAHASISWA BERBASIS PENGENALAN WAJAH
Versi Streamlit - Universitas Muhammadiyah Makassar
Menggunakan DeepFace (tidak butuh dlib/cmake)
"""

import streamlit as st
import cv2
import pickle
import os
import csv
import numpy as np
import hashlib
from datetime import datetime
from PIL import Image
import io

# Gunakan DeepFace untuk akurasi tinggi tanpa dlib
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# ── Konfigurasi Halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sistem Absensi - UNISMUH",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Konstanta ─────────────────────────────────────────────────────────────────
DB_FILE    = "database_wajah.pkl"
REKAP_FILE = "rekap_absensi.csv"
USERS_FILE = "users_sistem.pkl"
DATA_DIR   = "data_wajah"
os.makedirs(DATA_DIR, exist_ok=True)

DAFTAR_MATKUL = [
    "Pemrograman Web", "Basis Data", "Algoritma & Pemrograman",
    "Jaringan Komputer", "Kecerdasan Buatan", "Rekayasa Perangkat Lunak",
    "Sistem Operasi", "Matematika Diskrit", "Struktur Data", "Pemrograman Mobile",
]
DAFTAR_KELAS = ["A", "B", "C", "D", "E"]

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── CSS Kustom ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        padding: 20px 30px; border-radius: 12px;
        border: 1px solid #30363d; margin-bottom: 20px;
    }
    .main-header h1 { color: #58a6ff; margin: 0; font-size: 24px; }
    .main-header p  { color: #8b949e; margin: 4px 0 0; font-size: 13px; }
    .stat-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 16px; text-align: center;
    }
    .stat-card .num { font-size: 32px; font-weight: bold; color: #58a6ff; }
    .stat-card .lbl { font-size: 12px; color: #8b949e; margin-top: 4px; }
    .sukses-box {
        background: #0d2818; border: 1px solid #00e676;
        border-radius: 8px; padding: 16px; text-align: center;
        color: #00e676; font-size: 18px; font-weight: bold;
    }
    .gagal-box {
        background: #2d0a0a; border: 1px solid #ff5252;
        border-radius: 8px; padding: 16px; text-align: center;
        color: #ff5252; font-size: 16px;
    }
    .info-box {
        background: #0d1f38; border: 1px solid #58a6ff;
        border-radius: 8px; padding: 12px; color: #58a6ff; font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ── Fungsi Utilitas ───────────────────────────────────────────────────────────
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def muat_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    users = {
        "admin": {"password": hash_password("admin123"), "role": "admin", "nama": "Administrator"},
        "dosen": {"password": hash_password("dosen123"), "role": "dosen",  "nama": "Dosen"},
    }
    with open(USERS_FILE, "wb") as f:
        pickle.dump(users, f)
    return users

def cek_login(username, password):
    users = muat_users()
    if username in users:
        if users[username]["password"] == hash_password(password):
            return users[username]
    return None

def muat_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def simpan_database(db):
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)

def ambil_fitur_wajah_deepface(image_path):
    """Ekstrak embedding 512-dimensi menggunakan DeepFace (Facenet512)."""
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="opencv"
        )
        if result and len(result) > 0:
            return np.array(result[0]["embedding"])
    except Exception as e:
        st.error(f"DeepFace error: {e}")
    return None

def ambil_fitur_wajah_fallback(gray_roi):
    """Fallback: LBPH histogram."""
    roi = cv2.resize(gray_roi, (128, 128))
    roi = cv2.equalizeHist(roi)
    lbph = np.zeros((256,), dtype=np.float64)
    for i in range(1, roi.shape[0]-1):
        for j in range(1, roi.shape[1]-1):
            center = roi[i, j]
            code = 0
            neighbors = [
                roi[i-1,j-1], roi[i-1,j], roi[i-1,j+1],
                roi[i,j+1], roi[i+1,j+1], roi[i+1,j],
                roi[i+1,j-1], roi[i,j-1]
            ]
            for k, n in enumerate(neighbors):
                if n >= center:
                    code |= (1 << k)
            lbph[code] += 1
    lbph = lbph / (lbph.sum() + 1e-6)
    return lbph

def cocokkan_wajah(encoding_baru, database):
    """
    DeepFace: threshold euclidean distance <= 200 (Facenet512)
    Fallback: cosine similarity >= 0.92
    """
    best_nim, best_nama, best_skor = None, None, 0.0
    is_deep = DEEPFACE_AVAILABLE and len(encoding_baru) == 512

    for nim, data in database.items():
        if not isinstance(data, dict) or "encoding" not in data:
            continue
        enc_db = data["encoding"]
        if enc_db is None:
            continue
        try:
            if is_deep and len(enc_db) == 512:
                jarak = float(np.linalg.norm(encoding_baru - enc_db))
                skor = max(0.0, 1.0 - jarak / 400.0)
                cocok = jarak <= 200.0
            else:
                a = encoding_baru / (np.linalg.norm(encoding_baru) + 1e-6)
                b = enc_db / (np.linalg.norm(enc_db) + 1e-6)
                skor = float(np.dot(a, b))
                cocok = skor >= 0.92
            if skor > best_skor:
                best_skor = skor
                if cocok:
                    best_nim  = nim
                    best_nama = data.get("nama", nim)
        except Exception:
            continue

    return best_nim, best_nama, best_skor

def catat_absensi(nim, nama, matkul, kelas):
    sekarang = datetime.now()
    tanggal  = sekarang.strftime("%Y-%m-%d")
    waktu    = sekarang.strftime("%H:%M:%S")
    if os.path.exists(REKAP_FILE):
        with open(REKAP_FILE, "r") as f:
            for row in csv.reader(f):
                if len(row) >= 5 and row[0] == nim and row[2] == tanggal and row[4] == matkul:
                    return False, "sudah_absen"
    file_baru = not os.path.exists(REKAP_FILE)
    with open(REKAP_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if file_baru:
            writer.writerow(["NIM", "Nama", "Tanggal", "Waktu", "Mata Kuliah", "Kelas", "Status"])
        writer.writerow([nim, nama, tanggal, waktu, matkul, kelas, "HADIR"])
    return True, waktu

def deteksi_wajah_dari_gambar(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, None
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    return frame, gray, wajah

def simpan_gambar_sementara(image_bytes, nim):
    path  = os.path.join(DATA_DIR, f"temp_{nim}.jpg")
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite(path, frame)
    return path

# ── Inisialisasi Session State ────────────────────────────────────────────────
if "logged_in"  not in st.session_state: st.session_state.logged_in  = False
if "user_info"  not in st.session_state: st.session_state.user_info  = None
if "halaman"    not in st.session_state: st.session_state.halaman    = "dashboard"
if "absensi_ok" not in st.session_state: st.session_state.absensi_ok = None

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN LOGIN
# ══════════════════════════════════════════════════════════════════════════════
def halaman_login():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 30px 0 10px'>
            <div style='font-size:52px'>🎓</div>
            <h2 style='color:#58a6ff; margin:8px 0 4px'>Sistem Absensi Wajah</h2>
            <p style='color:#8b949e; font-size:13px'>Universitas Muhammadiyah Makassar</p>
        </div>
        """, unsafe_allow_html=True)

        tab_admin, tab_mhs = st.tabs(["👨‍💼 Admin / Dosen", "🎓 Mahasiswa"])

        with tab_admin:
            with st.form("form_login_admin"):
                username = st.text_input("Username", placeholder="admin / dosen")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submit   = st.form_submit_button("Masuk", use_container_width=True, type="primary")
            if submit:
                user = cek_login(username, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_info = {**user, "username": username}
                    st.session_state.halaman   = "rekap" if user["role"] == "dosen" else "dashboard"
                    st.rerun()
                else:
                    st.error("Username atau password salah!")

        with tab_mhs:
            with st.form("form_login_mhs"):
                nim_input  = st.text_input("NIM", placeholder="contoh: 105841100121")
                submit_mhs = st.form_submit_button("Masuk", use_container_width=True, type="primary")
            if submit_mhs:
                db = muat_database()
                if nim_input.strip() in db:
                    data_mhs = db[nim_input.strip()]
                    nama_mhs = data_mhs.get("nama", nim_input) if isinstance(data_mhs, dict) else nim_input
                    st.session_state.logged_in = True
                    st.session_state.user_info = {
                        "username": nim_input.strip(),
                        "nama": nama_mhs,
                        "role": "mahasiswa",
                        "nim": nim_input.strip()
                    }
                    st.session_state.halaman = "absensi"
                    st.rerun()
                else:
                    st.error("NIM tidak ditemukan! Hubungi admin untuk mendaftar.")

        st.markdown("""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:8px;
                    padding:12px; margin-top:16px; font-size:12px; color:#8b949e'>
            <b style='color:#ffd740'>Default login:</b><br>
            Admin → admin / admin123<br>
            Dosen → dosen / dosen123<br>
            Mahasiswa → masuk dengan NIM yang sudah terdaftar
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGASI
# ══════════════════════════════════════════════════════════════════════════════
def sidebar_navigasi():
    with st.sidebar:
        user = st.session_state.user_info
        role = user.get("role", "")
        ikon_role = "🎓" if role == "mahasiswa" else "👨‍🏫" if role == "dosen" else "⚙️"
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:10px;
                    padding:14px; margin-bottom:16px; text-align:center'>
            <div style='font-size:32px'>{ikon_role}</div>
            <div style='color:#e6edf3; font-weight:bold'>{user['nama']}</div>
            <div style='color:#8b949e; font-size:12px'>{role.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Menu Utama**")

        if role == "mahasiswa":
            menus = [("📸", "Absensi Wajah", "absensi")]
        elif role == "dosen":
            menus = [("📊", "Rekap Absensi", "rekap")]
        else:
            menus = [
                ("🏠", "Dashboard",        "dashboard"),
                ("📸", "Absensi Wajah",    "absensi"),
                ("👤", "Daftar Mahasiswa", "daftar"),
                ("📊", "Rekap Absensi",    "rekap"),
                ("⚙️", "Manajemen User",   "user"),
            ]

        for icon, label, key in menus:
            aktif = st.session_state.halaman == key
            if st.button(f"{icon} {label}", use_container_width=True,
                         type="primary" if aktif else "secondary"):
                st.session_state.halaman = key
                st.rerun()

        st.divider()
        if st.button("🚪 Keluar", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.session_state.halaman   = "dashboard"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def halaman_dashboard():
    st.markdown("""
    <div class='main-header'>
        <h1>🏠 Dashboard</h1>
        <p>Selamat datang di Sistem Absensi Berbasis Pengenalan Wajah</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Data", type="secondary"):
        st.rerun()

    import pandas as pd
    db = muat_database()
    total_mhs        = len(db)
    tanggal_hari_ini = datetime.now().strftime("%Y-%m-%d")
    total_absensi    = 0
    hadir_hari_ini   = 0
    df_rekap         = None

    if os.path.exists(REKAP_FILE):
        try:
            df_rekap = pd.read_csv(REKAP_FILE, dtype=str).fillna("")
            if not df_rekap.empty:
                total_absensi = len(df_rekap)
                if "Tanggal" in df_rekap.columns:
                    hadir_hari_ini = len(df_rekap[df_rekap["Tanggal"] == tanggal_hari_ini])
        except Exception:
            df_rekap = None

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [
        (c1, total_mhs,           "Total Mahasiswa"),
        (c2, hadir_hari_ini,      "Hadir Hari Ini"),
        (c3, total_absensi,       "Total Absensi"),
        (c4, len(DAFTAR_MATKUL),  "Mata Kuliah"),
    ]:
        with col:
            st.markdown(f"""
            <div class='stat-card'>
                <div class='num'>{num}</div>
                <div class='lbl'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 📋 Absensi Hari Ini")
        if df_rekap is not None and not df_rekap.empty and "Tanggal" in df_rekap.columns:
            df_hari = df_rekap[df_rekap["Tanggal"] == tanggal_hari_ini]
            st.dataframe(df_hari, use_container_width=True, hide_index=True) if not df_hari.empty else st.info("Belum ada absensi hari ini.")
        else:
            st.info("Belum ada data absensi.")

    with col_b:
        st.markdown("#### 👤 Mahasiswa Terdaftar")
        if db:
            data_mhs = [
                {"NIM": v.get("nim", k), "Nama": v.get("nama", "-")} if isinstance(v, dict)
                else {"NIM": k, "Nama": str(v)}
                for k, v in db.items()
            ]
            st.dataframe(pd.DataFrame(data_mhs), use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada mahasiswa terdaftar.")

# ══════════════════════════════════════════════════════════════════════════════
# ABSENSI WAJAH
# ══════════════════════════════════════════════════════════════════════════════
def halaman_absensi():
    st.markdown("""
    <div class='main-header'>
        <h1>📸 Absensi Pengenalan Wajah</h1>
        <p>Upload foto atau ambil gambar untuk absensi otomatis</p>
    </div>
    """, unsafe_allow_html=True)

    if DEEPFACE_AVAILABLE:
        st.success("🤖 Menggunakan **DeepFace (Facenet512)** — akurasi tinggi, sulit ditipu.")
    else:
        st.warning("⚠️ DeepFace tidak terinstall. Tambahkan `deepface` ke `requirements.txt`. Saat ini menggunakan metode LBPH.")

    db = muat_database()
    if not db:
        st.warning("⚠️ Database kosong! Daftarkan mahasiswa dulu di menu Daftar Mahasiswa.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### ⚙️ Pengaturan Sesi")
        matkul = st.selectbox("Mata Kuliah", DAFTAR_MATKUL)
        kelas  = st.selectbox("Kelas", DAFTAR_KELAS)

        st.markdown("#### 📷 Ambil Foto")
        sumber = st.radio("Sumber Gambar", ["Upload File", "Kamera Browser"], horizontal=True)

        foto_bytes = None
        if sumber == "Upload File":
            uploaded = st.file_uploader("Upload foto wajah", type=["jpg", "jpeg", "png"])
            if uploaded:
                foto_bytes = uploaded.read()
        else:
            foto_cam = st.camera_input("Ambil foto sekarang")
            if foto_cam:
                foto_bytes = foto_cam.read()

    with col2:
        st.markdown("#### 🔍 Hasil Pengenalan")

        if foto_bytes:
            frame, gray, wajah_list = deteksi_wajah_dari_gambar(foto_bytes)
            if frame is None:
                st.error("Gagal membaca gambar.")
                return

            tampilan = frame.copy()

            if len(wajah_list) == 0:
                st.markdown("<div class='gagal-box'>❌ Tidak ada wajah terdeteksi.</div>", unsafe_allow_html=True)
                st.image(Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB)), use_column_width=True)

            elif len(wajah_list) > 1:
                st.markdown("<div class='gagal-box'>⚠️ Lebih dari 1 wajah. Pastikan hanya 1 orang.</div>", unsafe_allow_html=True)
                for (x, y, fw, fh) in wajah_list:
                    cv2.rectangle(tampilan, (x,y), (x+fw,y+fh), (0,0,255), 2)
                st.image(Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB)), use_column_width=True)

            else:
                (x, y, fw, fh) = wajah_list[0]
                roi = gray[y:y+fh, x:x+fw]

                if DEEPFACE_AVAILABLE:
                    img_path = simpan_gambar_sementara(foto_bytes, "absensi_temp")
                    encoding = ambil_fitur_wajah_deepface(img_path)
                    if encoding is None:
                        encoding = ambil_fitur_wajah_fallback(roi)
                else:
                    encoding = ambil_fitur_wajah_fallback(roi)

                nim, nama, skor = cocokkan_wajah(encoding, db)

                if nim:
                    cv2.rectangle(tampilan, (x,y), (x+fw,y+fh), (0,255,0), 3)
                    cv2.putText(tampilan, f"{nama}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    st.image(Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB)), use_column_width=True)
                    st.info(f"🎯 Tingkat kecocokan: **{skor*100:.1f}%**")

                    berhasil, info = catat_absensi(nim, nama, matkul, kelas)
                    if berhasil:
                        st.markdown(f"""
                        <div class='sukses-box'>
                            ✅ ABSENSI BERHASIL!<br>
                            <span style='font-size:14px;color:#e6edf3'>{nama} ({nim})</span><br>
                            <span style='font-size:13px;color:#8b949e'>{matkul} - Kelas {kelas} | {info}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='info-box'>ℹ️ {nama} sudah absen hari ini.</div>", unsafe_allow_html=True)
                else:
                    cv2.rectangle(tampilan, (x,y), (x+fw,y+fh), (0,0,255), 3)
                    st.image(Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB)), use_column_width=True)
                    st.markdown(f"""
                    <div class='gagal-box'>
                        ❌ Wajah tidak dikenali<br>
                        <span style='font-size:13px'>Skor: {skor*100:.1f}% — pastikan pencahayaan cukup dan wajah menghadap kamera.</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#161b22; border:1px dashed #30363d; border-radius:10px;
                        padding:40px; text-align:center; color:#8b949e'>
                <div style='font-size:48px'>📷</div>
                <p>Upload foto atau gunakan kamera untuk memulai absensi</p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DAFTAR MAHASISWA
# ══════════════════════════════════════════════════════════════════════════════
def halaman_daftar():
    st.markdown("""
    <div class='main-header'>
        <h1>👤 Pendaftaran Wajah Mahasiswa</h1>
        <p>Daftarkan wajah mahasiswa ke dalam database sistem</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["➕ Daftarkan Mahasiswa Baru", "📋 Lihat Daftar Mahasiswa"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Data Mahasiswa")
            nama = st.text_input("Nama Lengkap", placeholder="contoh: Budi Santoso")
            nim  = st.text_input("NIM", placeholder="contoh: 105841100121")

            st.markdown("#### 📷 Foto Wajah")
            sumber = st.radio("Sumber Foto", ["Upload File", "Kamera Browser"], horizontal=True, key="sumber_daftar")
            foto_bytes = None
            if sumber == "Upload File":
                uploaded = st.file_uploader("Upload foto wajah", type=["jpg","jpeg","png"], key="upload_daftar")
                if uploaded:
                    foto_bytes = uploaded.read()
            else:
                foto_cam = st.camera_input("Ambil foto wajah", key="cam_daftar")
                if foto_cam:
                    foto_bytes = foto_cam.read()

        with col2:
            st.markdown("#### Preview & Konfirmasi")
            if foto_bytes:
                frame, gray, wajah_list = deteksi_wajah_dari_gambar(foto_bytes)
                if frame is not None:
                    tampilan = frame.copy()
                    for (x,y,fw,fh) in wajah_list:
                        cv2.rectangle(tampilan, (x,y), (x+fw,y+fh), (0,255,0), 2)
                    st.image(Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB)), use_column_width=True)
                    if len(wajah_list) > 0:
                        st.success(f"✅ {len(wajah_list)} wajah terdeteksi")
                    else:
                        st.error("❌ Tidak ada wajah terdeteksi. Ganti foto.")

            if st.button("💾 Simpan ke Database", type="primary", use_container_width=True):
                if not nama.strip() or not nim.strip():
                    st.error("Nama dan NIM tidak boleh kosong!")
                elif not foto_bytes:
                    st.error("Foto belum diupload!")
                else:
                    frame, gray, wajah_list = deteksi_wajah_dari_gambar(foto_bytes)
                    if frame is None or len(wajah_list) == 0:
                        st.error("Wajah tidak terdeteksi. Gunakan foto yang lebih jelas.")
                    elif len(wajah_list) > 1:
                        st.error("Lebih dari 1 wajah. Gunakan foto 1 orang saja.")
                    else:
                        nama_file = f"{nim}_{nama.replace(' ','_')}.jpg"
                        img_path  = os.path.join(DATA_DIR, nama_file)
                        cv2.imwrite(img_path, frame)

                        if DEEPFACE_AVAILABLE:
                            encoding = ambil_fitur_wajah_deepface(img_path)
                            metode   = "DeepFace Facenet512"
                        else:
                            (x,y,fw,fh) = wajah_list[0]
                            encoding = ambil_fitur_wajah_fallback(gray[y:y+fh, x:x+fw])
                            metode   = "LBPH fallback"

                        if encoding is None:
                            st.error("Gagal mengekstrak fitur wajah. Coba foto lain.")
                        else:
                            db = muat_database()
                            db[nim] = {"nama": nama.strip(), "nim": nim.strip(), "encoding": encoding}
                            simpan_database(db)
                            st.success(f"✅ **{nama}** ({nim}) berhasil didaftarkan dengan {metode}! Total: {len(db)} mahasiswa.")
                            st.balloons()

    with tab2:
        db = muat_database()
        st.markdown(f"#### Daftar Mahasiswa Terdaftar ({len(db)} orang)")
        if db:
            import pandas as pd
            data = []
            for i, (nim_key, v) in enumerate(db.items()):
                enc = v.get("encoding") if isinstance(v, dict) else None
                metode = "DeepFace" if enc is not None and isinstance(enc, np.ndarray) and len(enc) == 512 else "LBPH"
                if isinstance(v, dict):
                    data.append({"No": i+1, "NIM": v.get("nim", nim_key), "Nama": v.get("nama", "-"), "Metode": metode})
                else:
                    data.append({"No": i+1, "NIM": nim_key, "Nama": str(v), "Metode": "-"})
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### ❌ Hapus Mahasiswa")
            pilihan_hapus = [
                f"{v.get('nim', k)} - {v.get('nama', '-')}" if isinstance(v, dict) else k
                for k, v in db.items()
            ]
            nim_hapus = st.selectbox("Pilih mahasiswa yang akan dihapus", options=pilihan_hapus)
            if st.button("🗑️ Hapus dari Database", type="secondary"):
                nim_key = nim_hapus.split(" - ")[0].strip()
                if nim_key in db:
                    v = db[nim_key]
                    nama_hapus = v.get("nama", nim_key) if isinstance(v, dict) else str(v)
                    del db[nim_key]
                    simpan_database(db)
                    st.success(f"✅ {nama_hapus} berhasil dihapus.")
                    st.rerun()
        else:
            st.info("Belum ada mahasiswa terdaftar.")

# ══════════════════════════════════════════════════════════════════════════════
# REKAP ABSENSI
# ══════════════════════════════════════════════════════════════════════════════
def halaman_rekap():
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Rekap Absensi</h1>
        <p>Laporan dan data kehadiran mahasiswa</p>
    </div>
    """, unsafe_allow_html=True)

    role     = st.session_state.user_info.get("role", "admin")
    is_dosen = role == "dosen"

    if not os.path.exists(REKAP_FILE):
        st.info("Belum ada data absensi.")
        return

    import pandas as pd
    try:
        df = pd.read_csv(REKAP_FILE, dtype=str).fillna("")
    except Exception:
        st.error("File rekap rusak.")
        return

    if df.empty:
        st.info("Data absensi masih kosong.")
        return

    # Gunakan kolom standar agar tidak crash jika file CSV lama punya kolom berbeda
    header = ["NIM", "Nama", "Tanggal", "Waktu", "Mata Kuliah", "Kelas", "Status"]
    for col in header:
        if col not in df.columns:
            df[col] = ""
    df = df[header]

    if is_dosen:
        tab1, tab2, tab3 = st.tabs(["📋 Semua Data", "📅 Filter", "📥 Export"])
        tab4 = None
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Semua Data", "📅 Filter", "📥 Export", "🗑️ Hapus Data"])

    with tab1:
        st.markdown(f"#### Semua Rekap ({len(df)} catatan)")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.divider()
        if "NIM" in df.columns and "Nama" in df.columns:
            st.markdown("#### Ringkasan Per Mahasiswa")
            st.dataframe(df.groupby(["NIM","Nama"]).size().reset_index(name="Jumlah Hadir"),
                         use_container_width=True, hide_index=True)

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1: filter_tgl = st.date_input("Filter Tanggal", value=None)
        with col2: filter_mk  = st.selectbox("Mata Kuliah", ["Semua"] + DAFTAR_MATKUL)
        with col3: filter_kls = st.selectbox("Kelas", ["Semua"] + DAFTAR_KELAS)

        df_f = df.copy()
        if filter_tgl: df_f = df_f[df_f["Tanggal"] == str(filter_tgl)]
        if filter_mk  != "Semua" and "Mata Kuliah" in df_f.columns: df_f = df_f[df_f["Mata Kuliah"] == filter_mk]
        if filter_kls != "Semua" and "Kelas"       in df_f.columns: df_f = df_f[df_f["Kelas"]       == filter_kls]

        st.markdown(f"#### Hasil Filter ({len(df_f)} data)")
        st.dataframe(df_f, use_container_width=True, hide_index=True)

    with tab3:
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"),
                           f"rekap_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
                           use_container_width=True, type="primary")

    if tab4 is not None:
        with tab4:
            st.markdown("#### 🗑️ Hapus Data Rekap")
            st.warning("⚠️ Data yang dihapus tidak dapat dikembalikan!")
            hapus_mode = st.radio("Metode hapus:", [
                "Hapus per baris", "Hapus berdasarkan tanggal",
                "Hapus berdasarkan mahasiswa", "Hapus semua data"
            ])
            st.divider()

            if hapus_mode == "Hapus per baris":
                df_t = df.copy(); df_t.insert(0, "No", range(1, len(df_t)+1))
                st.dataframe(df_t, use_container_width=True, hide_index=True)
                idx = st.selectbox("Pilih baris:", range(len(df)),
                    format_func=lambda i: f"No.{i+1} | {df.iloc[i].get('NIM','?')} - {df.iloc[i].get('Nama','?')} | {df.iloc[i].get('Tanggal','?')}")
                if st.button("🗑️ Hapus Baris", type="secondary", use_container_width=True):
                    df.drop(index=idx).reset_index(drop=True).to_csv(REKAP_FILE, index=False)
                    st.success(f"✅ Baris No.{idx+1} dihapus.")
                    st.rerun()

            elif hapus_mode == "Hapus berdasarkan tanggal":
                tgl_list = sorted(df["Tanggal"].unique().tolist()) if "Tanggal" in df.columns else []
                if not tgl_list:
                    st.info("Tidak ada data.")
                else:
                    tgl = st.selectbox("Pilih tanggal:", tgl_list)
                    n   = len(df[df["Tanggal"] == tgl])
                    st.info(f"📌 {n} data pada {tgl} akan dihapus.")
                    if st.button(f"🗑️ Hapus Tanggal {tgl}", type="secondary", use_container_width=True):
                        df[df["Tanggal"] != tgl].reset_index(drop=True).to_csv(REKAP_FILE, index=False)
                        st.success(f"✅ {n} data tanggal {tgl} dihapus.")
                        st.rerun()

            elif hapus_mode == "Hapus berdasarkan mahasiswa":
                if "NIM" not in df.columns:
                    st.info("Kolom NIM tidak ada.")
                else:
                    mhs_list   = df.groupby(["NIM","Nama"]).size().reset_index()
                    pilih_mhs  = [f"{r['NIM']} - {r['Nama']}" for _, r in mhs_list.iterrows()]
                    mhs_hapus  = st.selectbox("Pilih mahasiswa:", pilih_mhs)
                    nim_h      = mhs_hapus.split(" - ")[0].strip()
                    n          = len(df[df["NIM"] == nim_h])
                    st.info(f"📌 {n} data {mhs_hapus} akan dihapus.")
                    if st.button(f"🗑️ Hapus {mhs_hapus}", type="secondary", use_container_width=True):
                        df[df["NIM"] != nim_h].reset_index(drop=True).to_csv(REKAP_FILE, index=False)
                        st.success(f"✅ {n} data dihapus.")
                        st.rerun()

            elif hapus_mode == "Hapus semua data":
                st.error("🚨 Akan menghapus SELURUH rekap absensi!")
                konfirmasi = st.text_input("Ketik HAPUS SEMUA untuk konfirmasi:")
                if st.button("🗑️ Hapus Semua", type="secondary", use_container_width=True):
                    if konfirmasi.strip() == "HAPUS SEMUA":
                        pd.DataFrame(columns=header).to_csv(REKAP_FILE, index=False)
                        st.success("✅ Semua rekap dihapus.")
                        st.rerun()
                    else:
                        st.error("❌ Konfirmasi salah!")

# ══════════════════════════════════════════════════════════════════════════════
# MANAJEMEN USER
# ══════════════════════════════════════════════════════════════════════════════
def halaman_user():
    st.markdown("""
    <div class='main-header'>
        <h1>⚙️ Manajemen User</h1>
        <p>Kelola akun pengguna sistem</p>
    </div>
    """, unsafe_allow_html=True)

    users = muat_users()
    import pandas as pd

    st.markdown(f"#### Daftar User ({len(users)} akun)")
    st.dataframe(pd.DataFrame([{"Username": k, "Nama": v["nama"], "Role": v["role"].upper()} for k, v in users.items()]),
                 use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### ➕ Tambah / Ubah User")
    col1, col2 = st.columns(2)
    with col1:
        new_user = st.text_input("Username baru")
        new_nama = st.text_input("Nama lengkap")
    with col2:
        new_pass = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", ["dosen", "admin"])

    if st.button("💾 Simpan User", type="primary"):
        if not new_user.strip() or not new_pass.strip() or not new_nama.strip():
            st.error("Semua field wajib diisi!")
        else:
            users[new_user.strip()] = {"password": hash_password(new_pass), "role": new_role, "nama": new_nama.strip()}
            with open(USERS_FILE, "wb") as f:
                pickle.dump(users, f)
            st.success(f"✅ User **{new_user}** berhasil disimpan.")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    halaman_login()
else:
    role = st.session_state.user_info.get("role", "admin")
    sidebar_navigasi()
    h = st.session_state.halaman

    if role == "mahasiswa":
        halaman_absensi()
    elif role == "dosen":
        halaman_rekap()
    else:
        if   h == "dashboard": halaman_dashboard()
        elif h == "absensi":   halaman_absensi()
        elif h == "daftar":    halaman_daftar()
        elif h == "rekap":     halaman_rekap()
        elif h == "user":      halaman_user()
