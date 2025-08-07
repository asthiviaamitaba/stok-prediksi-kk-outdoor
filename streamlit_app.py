import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ==== Load Model ====
model = joblib.load('decision_tree_model.pkl')
le_barang = joblib.load('le_barang.pkl')

st.title("📦 Prediksi Kebutuhan Stok Barang - KK Outdoor")
st.caption("Gunakan data historis untuk memprediksi kebutuhan stok per bulan")

uploaded_file = st.file_uploader("📁 Upload dataset stok (.csv)", type="csv")
if uploaded_file is None:
    st.warning("⚠️ Harap upload file CSV terlebih dahulu.")
    st.stop()

# Proses file
df = pd.read_csv(uploaded_file)
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df = df[df['Jumlah'].notnull()].copy()
df['Nama Barang'] = df['Nama Barang'].str.lower().str.strip()
df['Nama Barang'] = df['Nama Barang'].str.split(', ')
df = df.explode('Nama Barang').reset_index(drop=True)

df['Bulan'] = df['Tanggal'].dt.month
df['Barang_Encoded'] = le_barang.transform(df['Nama Barang'])

# Input bulan
st.subheader("🗓 Pilih Bulan Prediksi")
bulan = st.selectbox("Bulan", list(range(1, 13)))

barang_unik = sorted(df['Nama Barang'].unique())

# Prediksi
st.subheader("📊 Hasil Prediksi")
prediksi_data = pd.DataFrame({
    'Nama Barang': barang_unik,
    'Bulan': bulan
})
# Transform nama barang agar cocok dengan model
X_pred = prediksi_data.copy()
X_pred['Nama Barang'] = le_barang.transform(X_pred['Nama Barang'])

# Prediksi
prediksi_data['Jumlah Diprediksi'] = model.predict(X_pred).round()

st.dataframe(prediksi_data)

st.bar_chart(prediksi_data.set_index('Nama Barang')['Jumlah Diprediksi'])

st.download_button(
    label="⬇️ Download CSV Prediksi",
    data=prediksi_data.to_csv(index=False).encode('utf-8'),
    file_name=f'prediksi_stok_bulan_{bulan}.csv',
    mime='text/csv'
)
