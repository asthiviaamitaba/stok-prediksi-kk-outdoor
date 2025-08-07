import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ==== Load Model ====
model = joblib.load('decision_tree_model.pkl')
le_barang = joblib.load('le_barang.pkl')

# ==== UI Header ====
st.title("📦 Sistem Prediksi Kebutuhan Stok - Kula Outdoor")
st.write("Prediksi jumlah barang yang perlu disiapkan berdasarkan data historis peminjaman.")

# ==== Upload File CSV ====
uploaded_file = st.file_uploader("📁 Upload dataset stok (.csv)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("⚠️ Gunakan file bawaan karena belum ada file yang diupload.")
    st.stop()

# ==== Preprocessing ====
df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
df = df[df['Jumlah'].notnull()].copy()
df = df[df['Tanggal'].notnull()]

# Pisahkan nama barang yang digabung dengan koma
df['Nama_Barang'] = df['Nama Barang'].str.lower().str.strip()
df['Nama_Barang'] = df['Nama_Barang'].str.split(', ')
df = df.explode('Nama_Barang').reset_index(drop=True)

# Hapus spasi tambahan jika ada
df['Nama_Barang'] = df['Nama_Barang'].str.strip()

# Filter hanya barang yang dikenali encoder
barang_terdaftar = set(le_barang.classes_)
df = df[df['Nama_Barang'].isin(barang_terdaftar)]

# Tambah kolom Bulan & encoding barang
df['Bulan'] = df['Tanggal'].dt.month
df['Barang_Encoded'] = le_barang.transform(df['Nama_Barang'])

# ================================
# === FITUR TAMBAHAN: TOP BARANG
# ================================
st.subheader("🏆 Top Barang Paling Sering Dipinjam")

top_barang = df['Nama_Barang'].value_counts().reset_index()
top_barang.columns = ['Nama Barang', 'Jumlah Peminjaman']

col1, col2 = st.columns([1, 2])
with col1
