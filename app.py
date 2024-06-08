import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import string

# Load data
info_tourism = pd.read_csv("tourism_with_id.csv")

# Tab pertama: Filter Tempat Wisata
def filter_places():
    st.title('Filter Tempat Wisata')
    categories = st.selectbox('Category wisata?', info_tourism['Category'].unique())
    cities = st.selectbox('Lokasi?', info_tourism['City'].unique())
    selected_price_range = st.sidebar.slider('Range Harga?', min_value=min_price, max_value=max_price, value=(min_price, max_price))

    min_price, max_price = selected_price_range

    # Filter data berdasarkan input pengguna
    filtered_data = info_tourism[(info_tourism['Category'] == categories) &
                                 (info_tourism['City'] == cities) ]

    # Tampilkan hasil filter
    st.header('Tempat Wisata yang Sesuai dengan Preferensi Kamu')
    if len(filtered_data) == 0:
        st.write('Maaf, tidak ada tempat wisata yang sesuai dengan preferensi Kamu.')
    else:
        st.write(filtered_data[['Place_Name', 'Category', 'City', 'Rating']])
         
# Main App
st.title("Sistem Rekomendasi Tempat Wisata")

# Pilihan tab
tabs = ["Filter Tempat Wisata"]
choice = st.sidebar.radio("Navigasi", tabs)

# Tampilkan tab yang dipilih
if choice == "Filter Tempat Wisata":
    filter_places()
elif choice == "Rekomendasi berdasarkan Deskripsi":
    recommend_by_description()
