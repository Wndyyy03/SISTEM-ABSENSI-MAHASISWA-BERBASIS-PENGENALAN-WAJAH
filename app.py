"""
SISTEM ABSENSI MAHASISWA BERBASIS PENGENALAN WAJAH
Versi Streamlit - Universitas Muhammadiyah Makassar
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

def ambil_fitur_wajah(gray_roi):
    roi = cv2.resize(gray_roi, (128, 128))
    roi = cv2.equalizeHist(roi)
    return roi.flatten().astype(np.float64)

def cocokkan_wajah(encoding_baru, database, threshold=0.35):
    best_nim, best_nama, best_skor = None, None, -1
    for nim, data in database.items():
        enc_db = data["encoding"]
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
    """Deteksi wajah dari gambar yang diupload / difoto."""
    nparr  = np.frombuffer(image_bytes, np.uint8)
    frame  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None, None, None
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah  = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    return frame, gray, wajah

def gambar_ke_bytes(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

# ── Inisialisasi Session State ────────────────────────────────────────────────
if "logged_in"   not in st.session_state: st.session_state.logged_in   = False
if "user_info"   not in st.session_state: st.session_state.user_info   = None
if "halaman"     not in st.session_state: st.session_state.halaman     = "dashboard"
if "absensi_ok"  not in st.session_state: st.session_state.absensi_ok  = None

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

        with st.form("form_login"):
            username = st.text_input("Username", placeholder="admin / dosen")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submit   = st.form_submit_button("Masuk", use_container_width=True, type="primary")

        if submit:
            user = cek_login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_info = {**user, "username": username}
                st.rerun()
            else:
                st.error("Username atau password salah!")

        st.markdown("""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:8px;
                    padding:12px; margin-top:16px; font-size:12px; color:#8b949e'>
            <b style='color:#ffd740'>Default login:</b><br>
            Admin &nbsp;→ admin / admin123<br>
            Dosen &nbsp;→ dosen / dosen123
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGASI
# ══════════════════════════════════════════════════════════════════════════════
def sidebar_navigasi():
    with st.sidebar:
        user = st.session_state.user_info
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:10px;
                    padding:14px; margin-bottom:16px; text-align:center'>
            <div style='font-size:32px'>👤</div>
            <div style='color:#e6edf3; font-weight:bold'>{user['nama']}</div>
            <div style='color:#8b949e; font-size:12px'>{user['role'].upper()}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Menu Utama**")
        menus = [
            ("🏠", "Dashboard",         "dashboard"),
            ("📸", "Absensi Wajah",     "absensi"),
            ("👤", "Daftar Mahasiswa",  "daftar"),
            ("📊", "Rekap Absensi",     "rekap"),
        ]
        if user["role"] == "admin":
            menus.append(("⚙️", "Manajemen User", "user"))

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

    db     = muat_database()
    total_mhs = len(db)

    total_absensi = 0
    hadir_hari_ini = 0
    tanggal_hari_ini = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(REKAP_FILE):
        with open(REKAP_FILE, "r") as f:
            rows = list(csv.reader(f))[1:]
        total_absensi  = len(rows)
        hadir_hari_ini = sum(1 for r in rows if len(r) >= 3 and r[2] == tanggal_hari_ini)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl in [
        (c1, total_mhs,    "Total Mahasiswa"),
        (c2, hadir_hari_ini, "Hadir Hari Ini"),
        (c3, total_absensi,  "Total Absensi"),
        (c4, len(DAFTAR_MATKUL), "Mata Kuliah"),
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
        if os.path.exists(REKAP_FILE):
            with open(REKAP_FILE, "r") as f:
                rows = list(csv.reader(f))
            header = rows[0] if rows else []
            data_hari = [r for r in rows[1:] if len(r) >= 3 and r[2] == tanggal_hari_ini]
            if data_hari:
                import pandas as pd
                df = pd.DataFrame(data_hari, columns=header[:len(data_hari[0])])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Belum ada absensi hari ini.")
        else:
            st.info("Belum ada data absensi.")

    with col_b:
        st.markdown("#### 👤 Mahasiswa Terdaftar")
        if db:
            import pandas as pd
            data_mhs = [{"NIM": v["nim"], "Nama": v["nama"]} for v in db.values()]
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

    db = muat_database()
    if not db:
        st.warning("⚠️ Database wajah kosong! Daftarkan mahasiswa dulu di menu **Daftar Mahasiswa**.")
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
                st.error("Gagal membaca gambar. Coba lagi.")
                return

            tampilan = frame.copy()

            if len(wajah_list) == 0:
                st.markdown("<div class='gagal-box'>❌ Tidak ada wajah terdeteksi.<br>Pastikan wajah terlihat jelas dan cahaya cukup.</div>",
                            unsafe_allow_html=True)
                img_show = Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB))
                st.image(img_show, use_column_width=True)

            elif len(wajah_list) > 1:
                st.markdown("<div class='gagal-box'>⚠️ Lebih dari 1 wajah terdeteksi.<br>Pastikan hanya 1 orang dalam foto.</div>",
                            unsafe_allow_html=True)
                for (x, y, fw, fh) in wajah_list:
                    cv2.rectangle(tampilan, (x,y), (x+fw,y+fh), (0,0,255), 2)
                img_show = Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB))
                st.image(img_show, use_column_width=True)

            else:
                (x, y, fw, fh) = wajah_list[0]
                roi = gray[y:y+fh, x:x+fw]
                encoding = ambil_fitur_wajah(roi)
                nim, nama, skor = cocokkan_wajah(encoding, db)

                if nim:
                    cv2.rectangle(tampilan, (x,y), (x+fw,y+fh), (0,255,0), 3)
                    cv2.putText(tampilan, f"{nama}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    img_show = Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB))
                    st.image(img_show, use_column_width=True)

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
                        st.markdown(f"""
                        <div class='info-box'>
                            ℹ️ {nama} sudah absen hari ini untuk mata kuliah ini.
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    cv2.rectangle(tampilan, (x,y), (x+fw,y+fh), (0,0,255), 3)
                    img_show = Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB))
                    st.image(img_show, use_column_width=True)
                    st.markdown(f"""
                    <div class='gagal-box'>
                        ❌ Wajah tidak dikenali<br>
                        <span style='font-size:13px'>Skor kemiripan: {skor:.2f} (minimum 0.35)</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#161b22; border:1px dashed #30363d; border-radius:10px;
                        padding:40px; text-align:center; color:#8b949e'>
                <div style='font-size:48px'>📷</div>
                <p>Upload foto atau gunakan kamera browser<br>untuk memulai absensi</p>
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
                uploaded = st.file_uploader("Upload foto wajah yang jelas", type=["jpg","jpeg","png"], key="upload_daftar")
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
                    if len(wajah_list) > 0:
                        for (x,y,fw,fh) in wajah_list:
                            cv2.rectangle(tampilan, (x,y), (x+fw,y+fh), (0,255,0), 2)
                        st.image(Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB)),
                                 use_column_width=True)
                        st.success(f"✅ {len(wajah_list)} wajah terdeteksi")
                    else:
                        st.image(Image.fromarray(cv2.cvtColor(tampilan, cv2.COLOR_BGR2RGB)),
                                 use_column_width=True)
                        st.error("❌ Tidak ada wajah terdeteksi. Ganti foto.")

            if st.button("💾 Simpan ke Database", type="primary", use_container_width=True):
                if not nama.strip() or not nim.strip():
                    st.error("Nama dan NIM tidak boleh kosong!")
                elif not foto_bytes:
                    st.error("Foto belum diupload!")
                else:
                    frame, gray, wajah_list = deteksi_wajah_dari_gambar(foto_bytes)
                    if frame is None or len(wajah_list) == 0:
                        st.error("Wajah tidak terdeteksi dalam foto. Gunakan foto yang lebih jelas.")
                    elif len(wajah_list) > 1:
                        st.error("Lebih dari 1 wajah terdeteksi. Gunakan foto dengan 1 orang saja.")
                    else:
                        (x,y,fw,fh) = wajah_list[0]
                        roi = gray[y:y+fh, x:x+fw]
                        encoding = ambil_fitur_wajah(roi)

                        nama_file = f"{nim}_{nama.replace(' ','_')}.jpg"
                        cv2.imwrite(os.path.join(DATA_DIR, nama_file), frame)

                        db = muat_database()
                        db[nim] = {"nama": nama.strip(), "nim": nim.strip(), "encoding": encoding}
                        simpan_database(db)
                        st.success(f"✅ **{nama}** ({nim}) berhasil didaftarkan! Total: {len(db)} mahasiswa.")
                        st.balloons()

    with tab2:
        db = muat_database()
        st.markdown(f"#### Daftar Mahasiswa Terdaftar ({len(db)} orang)")
        if db:
            import pandas as pd
            data = []
            for i, (nim_key, v) in enumerate(db.items()):
                if isinstance(v, dict):
                    data.append({"No": i+1, "NIM": v.get("nim", nim_key), "Nama": v.get("nama", "-")})
                else:
                    data.append({"No": i+1, "NIM": nim_key, "Nama": str(v)})
            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### ❌ Hapus Mahasiswa")
            pilihan_hapus = []
            for nim_key, v in db.items():
                if isinstance(v, dict):
                    pilihan_hapus.append(f"{v.get('nim', nim_key)} - {v.get('nama', '-')}")
                else:
                    pilihan_hapus.append(nim_key)
            nim_hapus = st.selectbox("Pilih mahasiswa yang akan dihapus", options=pilihan_hapus)
            if st.button("🗑️ Hapus dari Database", type="secondary"):
                nim_key = nim_hapus.split(" - ")[0]
                if nim_key in db:
                    nama_hapus = db[nim_key]["nama"]
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

    if not os.path.exists(REKAP_FILE):
        st.info("Belum ada data absensi.")
        return

    import pandas as pd
    with open(REKAP_FILE, "r") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        st.info("Data absensi masih kosong.")
        return

    header = rows[0]
    df = pd.DataFrame(rows[1:], columns=header)

    tab1, tab2, tab3 = st.tabs(["📋 Semua Data", "📅 Filter", "📥 Export"])

    with tab1:
        st.markdown(f"#### Semua Rekap Absensi ({len(df)} catatan)")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Ringkasan Per Mahasiswa")
        if "NIM" in df.columns and "Nama" in df.columns:
            ringkasan = df.groupby(["NIM","Nama"]).size().reset_index(name="Jumlah Hadir")
            st.dataframe(ringkasan, use_container_width=True, hide_index=True)

    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_tgl = st.date_input("Filter Tanggal", value=None)
        with col2:
            if "Mata Kuliah" in df.columns:
                filter_mk = st.selectbox("Filter Mata Kuliah", ["Semua"] + DAFTAR_MATKUL)
            else:
                filter_mk = "Semua"
        with col3:
            if "Kelas" in df.columns:
                filter_kls = st.selectbox("Filter Kelas", ["Semua"] + DAFTAR_KELAS)
            else:
                filter_kls = "Semua"

        df_filter = df.copy()
        if filter_tgl:
            df_filter = df_filter[df_filter["Tanggal"] == str(filter_tgl)]
        if filter_mk != "Semua" and "Mata Kuliah" in df_filter.columns:
            df_filter = df_filter[df_filter["Mata Kuliah"] == filter_mk]
        if filter_kls != "Semua" and "Kelas" in df_filter.columns:
            df_filter = df_filter[df_filter["Kelas"] == filter_kls]

        st.markdown(f"#### Hasil Filter ({len(df_filter)} data)")
        st.dataframe(df_filter, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### Download Data Absensi")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"rekap_absensi_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )

# ══════════════════════════════════════════════════════════════════════════════
# MANAJEMEN USER (Admin only)
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
    data_user = [{"Username": k, "Nama": v["nama"], "Role": v["role"].upper()}
                 for k, v in users.items()]
    st.dataframe(pd.DataFrame(data_user), use_container_width=True, hide_index=True)

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
            users[new_user.strip()] = {
                "password": hash_password(new_pass),
                "role": new_role,
                "nama": new_nama.strip()
            }
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
    sidebar_navigasi()
    h = st.session_state.halaman
    if   h == "dashboard": halaman_dashboard()
    elif h == "absensi":   halaman_absensi()
    elif h == "daftar":    halaman_daftar()
    elif h == "rekap":     halaman_rekap()
    elif h == "user":      halaman_user()
