import streamlit as st
import joblib
import numpy as np

# Load models dan encoders
model_reg = joblib.load('decision_tree_model.pkl')
model_cls = joblib.load('naive_bayes_model.pkl')
le_barang = joblib.load('le_barang.pkl')
le_toko = joblib.load('le_toko.pkl')
le_kat = joblib.load('le_kat.pkl')

# Sidebar Input
st.title("🏕️ Prediksi Stok Barang")

nama_barang = st.selectbox("Pilih Nama Barang", le_barang.classes_)
toko = st.selectbox("Pilih Toko/Pelanggan", le_toko.classes_)
bulan = st.selectbox("Pilih Bulan", list(range(1, 13)))

# Transform input
barang_encoded = le_barang.transform([nama_barang])[0]
toko_encoded = le_toko.transform([toko])[0]

input_array = np.array([[barang_encoded, toko_encoded, bulan]])

# Prediksi regresi
prediksi_jumlah = model_reg.predict(input_array)[0]

# Prediksi klasifikasi
prediksi_kategori = le_kat.inverse_transform(model_cls.predict(input_array))[0]

# Output
st.subheader("Hasil Prediksi")
st.write(f"Prediksi jumlah stok: **{int(prediksi_jumlah)}**")
st.write(f"Kategori kebutuhan: **{prediksi_kategori}**")
