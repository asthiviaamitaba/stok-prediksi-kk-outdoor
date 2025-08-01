import streamlit as st
import pandas as pd

# ------------------------
# LOAD & SIAPKAN DATA
# ------------------------
data_url = "simpan_dataset_stok_alat_sewa (1).csv"
df = pd.read_csv(data_url)
df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors='coerce')
df["Bulan"] = df["Tanggal"].dt.month

# Pisahkan nama barang dan normalkan ke baris terpisah
df = df.dropna(subset=["Nama Barang"])
df["Nama Barang"] = df["Nama Barang"].str.split(",")
df = df.explode("Nama Barang")
df["Nama Barang"] = df["Nama Barang"].str.strip()

# Harga sewa per item (dari gambar harga sewa)
harga_sewa = {
    "Tenda": 50000,
    "Matras": 25000,
    "Kompor": 20000,
    "Lampu": 10000,
    "Kursi": 20000,
    "Gas Rent": 20000,
    "Gas Tukar": 22000,
    "Sarung Tangan": 15000,
    "Sleeping Bag": 30000,
    "Carrier": 60000,
    "Jaket": 25000,
    "Flysheet": 20000,
    "Headlamp": 10000,
    "Nest": 15000,
    "Tali": 5000,
    "Panci": 15000,
    "Trekking Pole": 25000,
    "Kompor Portable": 20000,
    "Cover Bag": 10000,
    "Trash Bag": 5000
}

# Estimasi pemasukan
df["Harga Sewa"] = df["Nama Barang"].map(harga_sewa)
df["Estimasi Pemasukan"] = df["Jumlah"] * df["Harga Sewa"]

# Agregasi data per bulan dan barang
agg_df = df.groupby(["Bulan", "Nama Barang"], as_index=False).agg({
    "Jumlah": "sum",
    "Estimasi Pemasukan": "sum"
})
agg_df.rename(columns={"Jumlah": "Total Disewa"}, inplace=True)

# ------------------------
# STREAMLIT APP
# ------------------------
st.title("📦 Prediksi Barang Paling Diminati")
st.write("Tentukan bulan untuk melihat barang yang paling banyak diminati serta estimasi pemasukan.")

bulan = st.selectbox("Pilih Bulan", sorted(agg_df["Bulan"].unique()))

top_n = 3
filtered = agg_df[agg_df["Bulan"] == bulan]
top_items = filtered.sort_values(by="Total Disewa", ascending=False).head(top_n)

st.subheader(f"Top {top_n} Barang Paling Diminati Bulan {bulan}")

for i, row in top_items.iterrows():
    st.markdown(f"**{int(i+1)}. {row['Nama Barang']}**")
    st.write(f"Jumlah Disewa: {int(row['Total Disewa'])}")
    st.write(f"Estimasi Pemasukan: Rp {int(row['Estimasi Pemasukan']):,}")
    st.markdown("---")

if top_items.empty:
    st.warning("Belum ada data untuk bulan ini.")
