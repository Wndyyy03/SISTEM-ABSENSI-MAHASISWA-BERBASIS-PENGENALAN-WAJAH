"""
==============================================
  LANGKAH 3: LIHAT REKAP ABSENSI
  Tampilkan laporan kehadiran mahasiswa
==============================================
"""

import csv
import os
from datetime import datetime

REKAP_FILE = "rekap_absensi.csv"

def lihat_rekap():
    if not os.path.exists(REKAP_FILE):
        print("\n⚠️  Belum ada data absensi.")
        print("   Jalankan dulu: python 2_absensi.py\n")
        return

    with open(REKAP_FILE, "r") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        print("\n⚠️  Data absensi masih kosong.\n")
        return

    header = rows[0]
    data   = rows[1:]

    print("\n" + "="*65)
    print(f"  REKAP ABSENSI - Total: {len(data)} catatan")
    print("="*65)
    print(f"{'No':<5} {'NIM':<15} {'Nama':<25} {'Tanggal':<12} {'Waktu'}")
    print("-"*65)

    for i, row in enumerate(data, 1):
        if len(row) >= 5:
            print(f"{i:<5} {row[0]:<15} {row[1]:<25} {row[2]:<12} {row[3]}")

    print("="*65)

    # Rekap per mahasiswa
    print("\n📊 REKAP KEHADIRAN PER MAHASISWA:")
    print("-"*40)
    rekap = {}
    for row in data:
        if len(row) >= 2:
            nim  = row[0]
            nama = row[1]
            rekap[nim] = rekap.get(nim, {"nama": nama, "hadir": 0})
            rekap[nim]["hadir"] += 1

    for nim, info in rekap.items():
        print(f"  {info['nama']:<25} → {info['hadir']} kali hadir")

    print()

def filter_tanggal():
    tanggal = input("\nMasukkan tanggal (YYYY-MM-DD): ").strip()

    if not os.path.exists(REKAP_FILE):
        print("⚠️  File rekap belum ada.")
        return

    with open(REKAP_FILE, "r") as f:
        rows = list(csv.reader(f))

    data = [r for r in rows[1:] if len(r) >= 3 and r[2] == tanggal]

    if not data:
        print(f"⚠️  Tidak ada data absensi pada tanggal {tanggal}")
        return

    print(f"\n{'='*55}")
    print(f"  ABSENSI TANGGAL: {tanggal}  ({len(data)} mahasiswa hadir)")
    print(f"{'='*55}")
    print(f"{'No':<5} {'NIM':<15} {'Nama':<25} {'Waktu'}")
    print("-"*55)
    for i, row in enumerate(data, 1):
        print(f"{i:<5} {row[0]:<15} {row[1]:<25} {row[3]}")
    print()


if __name__ == "__main__":
    while True:
        print("\n" + "="*40)
        print("  MENU REKAP ABSENSI")
        print("="*40)
        print("  1. Lihat semua rekap")
        print("  2. Filter berdasarkan tanggal")
        print("  3. Keluar")
        print("="*40)

        pilih = input("Pilihan (1/2/3): ").strip()

        if pilih == "1":
            lihat_rekap()
        elif pilih == "2":
            filter_tanggal()
        elif pilih == "3":
            print("👋 Keluar.")
            break
        else:
            print("⚠️  Pilihan tidak valid.")
