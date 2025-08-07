
import streamlit as st
import pandas as pd
import joblib
import io

# Set title and description
st.set_page_config(page_title="Prediksi Kebutuhan Stok - KK Outdoor", layout="wide")
st.title("📦 Prediksi Kebutuhan Stok - KK Outdoor")
st.markdown("Aplikasi ini memprediksi jumlah kebutuhan stok barang mingguan/bulanan berdasarkan data peminjaman sebelumnya menggunakan model Machine Learning (Decision Tree dan Naive Bayes).")

# Upload data file
uploaded_file = st.file_uploader("Unggah file CSV dataset peminjaman", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Validasi kolom
    expected_columns = ['Tanggal', 'Nama Barang', 'Jumlah']
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Dataset kamu kekurangan kolom: {missing_cols}")
        st.stop()

    # Preview
    st.subheader("📄 Preview Dataset")
    st.dataframe(df.head())

    # Preprocessing
    df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
    df.dropna(subset=['Tanggal'], inplace=True)
    df['Bulan'] = df['Tanggal'].dt.month

    # Load label encoder
    le_barang = joblib.load("le_barang.pkl")
    df['Nama Barang'] = df['Nama Barang'].astype(str).str.lower().str.strip()
    df['Barang_Encoded'] = le_barang.transform(df['Nama Barang'])

    # Pilih bulan untuk prediksi
    selected_month = st.selectbox("Pilih Bulan untuk Prediksi", sorted(df['Bulan'].unique()))
    bulan_data = df[df['Bulan'] == selected_month]

    if bulan_data.empty:
        st.warning("Tidak ada data untuk bulan yang dipilih.")
        st.stop()

    # Pilih model
    model_choice = st.radio("Pilih Model Prediksi", ["Decision Tree", "Naive Bayes"])
    if model_choice == "Decision Tree":
        model = joblib.load("decision_tree_model.pkl")
    else:
        model = joblib.load("naive_bayes_model.pkl")

    # Prediksi
    barang_grouped = bulan_data.groupby('Nama Barang').agg({'Jumlah': 'sum'}).reset_index()
    barang_grouped['Barang_Encoded'] = le_barang.transform(barang_grouped['Nama Barang'])
    barang_grouped['Bulan'] = selected_month
    X_pred = barang_grouped[['Barang_Encoded', 'Bulan']]
    y_pred = model.predict(X_pred)

    prediksi_data = barang_grouped.copy()
    prediksi_data['Jumlah Diprediksi'] = y_pred

    # Tabs UI
    tab1, tab2 = st.tabs(["📊 Hasil Prediksi", "📈 Visualisasi"])

    with tab1:
        st.subheader("📊 Tabel Prediksi Kebutuhan Stok")
        st.dataframe(prediksi_data[['Nama Barang', 'Jumlah Diprediksi']])

        # Download hasil prediksi
        csv = prediksi_data[['Nama Barang', 'Jumlah Diprediksi']].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Unduh Hasil Prediksi (CSV)", data=csv, file_name="prediksi_stok.csv", mime="text/csv")

    with tab2:
        st.subheader("📈 Visualisasi Jumlah Diprediksi")
        st.bar_chart(prediksi_data.set_index('Nama Barang')['Jumlah Diprediksi'])

    with st.expander("ℹ️ Tentang Model"):
        st.markdown("""
        - Model ini dibuat berdasarkan histori peminjaman alat di KK Outdoor.
        - Data dianalisis untuk melihat tren kebutuhan barang tiap bulan.
        - Model yang digunakan: **Decision Tree** dan **Naive Bayes**.
        """)
else:
    st.info("Silakan unggah file dataset untuk memulai prediksi.")
