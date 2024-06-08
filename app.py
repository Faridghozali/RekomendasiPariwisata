import streamlit as st
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load data
info_tourism = pd.read_csv("tourism_with_id.csv")

# CSS for background images
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://islamic-center.or.id/wp-content/uploads/2016/07/Pariwisata-Halal-Indonesia.jpg");
    background-size: cover;
    background-position: center;
}

.stApp > header {
    background-color: rgba(0,0,0,0);
}

.css-1d391kg {
    background-image: url("https://example.com/background_sidebar.jpg");
    background-size: cover;
    background-position: center;
}

/* Font color to black */
body, .css-10trblm, .css-1v3fvcr, .stText, .stNumberInput, .stSelectbox {
    color: black;
}

/* Selectbox background to white */
.css-1d391kg, .css-2tvg0n, .css-1a32fsj, .css-2hb7k5, .stSelectbox>div {
    background-color: white;
}


</style>
'''

# Apply CSS
st.markdown(page_bg_img, unsafe_allow_html=True)

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
