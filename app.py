import streamlit as st
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load data
info_tourism = pd.read_csv("tourism_with_id.csv")

# Tab pertama: Filter Tempat Wisata
def filter_places():
    # Input user for name and age
    name = st.text_input('Masukkan nama kamu:')
    age = st.number_input('Masukkan umur kamu:', min_value=10, max_value=100)
    
    categories = st.selectbox('Masukkan kategori wisata', info_tourism['Category'].unique())
    cities = st.selectbox('Masukan Lokasi kamu', info_tourism['City'].unique())

    # Tampilkan hasil filter hanya jika semua inputan sudah terisi
    if name and age and categories and cities:
        # Filter data berdasarkan input pengguna
        filtered_data = info_tourism[(info_tourism['Category'] == categories) &
                                     (info_tourism['City'] == cities)]

        st.header(f'Daftar rekomendasi wisata untuk {name} yang berumur {age} tahun :')

        if len(filtered_data) == 0:
            st.write('Mohon maaf, tidak ada rekomendasi tempat wisata yang sesuai dengan preferensi Kamu saat ini.')
        else:
            st.write(filtered_data[['Place_Name', 'Category', 'City', 'Price', 'Rating']])
    else:
        st.write('Silakan lengkapi semua input untuk melihat rekomendasi tempat wisata.')

# Main App
st.title("Rekomendasi Tempat Wisata di Indonesia")

# Pilihan tab
tabs = ["Sistem Rekomendasi Wisata"]
choice = st.sidebar.radio("Pilihan Menu", tabs)

# Tampilkan tab yang dipilih
if choice == "Sistem Rekomendasi Wisata":
    filter_places()
if choice == "Sistem Rekomendasi Wisata dari User sebelumnya":
    filter_places()
elif choice == "Visualisasi Data":
    recommend_by_description()
