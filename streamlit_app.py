import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ==== Load Model ====
model = joblib.load('decision_tree_model.pkl')
le_barang = joblib.load('le_barang.pkl')

# ==== UI Header ====
st.title("Sistem Prediksi Kebutuhan Stok KK Outdoor")
st.write("Prediksi jumlah barang yang perlu disiapkan berdasarkan data historis.")

# ==== Upload File CSV ====
uploaded_file = st.file_uploader("Upload dataset stok (.csv)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Gunakan file bawaan karena belum ada file yang diupload.")
    st.stop()

# ==== Preprocessing Sederhana ====
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df = df[df['Jumlah'].notnull()].copy()
df['Nama Barang'] = df['Nama Barang'].str.lower().str.strip()
df['Nama Barang'] = df['Nama Barang'].str.split(', ')
df = df.explode('Nama Barang').reset_index(drop=True)

df['Bulan'] = df['Tanggal'].dt.month
df['Barang_Encoded'] = le_barang.transform(df['Nama Barang'])

# ==== Pilih Periode Prediksi ====
st.subheader("Pilih Periode Prediksi")
bulan = st.selectbox("Bulan", list(range(1, 13)))
barang_unik = sorted(df['Nama Barang'].unique())

# ==== Prediksi Jumlah per Barang ====
st.subheader("Hasil Prediksi")
prediksi_data = pd.DataFrame({
    'Nama Barang': barang_unik,
    'Barang_Encoded': le_barang.transform(barang_unik),
    'Bulan': bulan
})
X_pred = prediksi_data[['Barang_Encoded', 'Bulan']]
prediksi_data['Jumlah Diprediksi'] = model.predict(X_pred).round()

# ==== Tampilkan Tabel ====
st.dataframe(prediksi_data[['Nama Barang', 'Jumlah Diprediksi']])

# ==== Chart ====
st.bar_chart(prediksi_data.set_index('Nama Barang')['Jumlah Diprediksi'])

# ==== Download CSV ====
st.download_button(
    label="Download Rekomendasi Stok (CSV)",
    data=prediksi_data.to_csv(index=False).encode('utf-8'),
    file_name='prediksi_stok.csv',
    mime='text/csv'
)
