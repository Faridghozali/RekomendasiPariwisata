import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load data
info_tourism = pd.read_csv("tourism_with_id.csv")
# Make sure the data for visualizations is loaded, e.g.:
# place = pd.read_csv("place_data.csv")
# rating = pd.read_csv("rating_data.csv")
# user = pd.read_csv("user_data.csv")

# CSS for background images and custom styling
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

/* Font color to black and bold */
body, .css-10trblm, .css-1v3fvcr, .stText, .stNumberInput, .stSelectbox {
    color: black;
    font-weight: bold;
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
    
    categories = st.selectbox('Kategori wisata', info_tourism['Category'].unique())
    cities = st.selectbox('Lokasi kamu', info_tourism['City'].unique())

    # Tampilkan hasil filter hanya jika semua inputan sudah terisi
    if name and age and categories and cities:
        # Filter data berdasarkan input pengguna
        filtered_data = info_tourism[(info_tourism['Category'] == categories) &
                                     (info_tourism['City'] == cities)]

        st.header(f'Daftar rekomendasi wisata untuk {name} yang berumur {age} tahun')

        if len(filtered_data) == 0:
            st.write('Mohon maaf, tidak ada rekomendasi tempat wisata yang sesuai dengan preferensi Kamu saat ini.')
        else:
            # Rename columns for display
            filtered_data_display = filtered_data.rename(columns={
                'Place_Name': 'Nama_Tempat',
                'Category': 'Kategori',
                'City': 'Lokasi',
                'Price': 'Harga',
                'Rating': 'Rating'
            })
            st.write(filtered_data_display[['Nama_Tempat', 'Kategori', 'Lokasi', 'Harga', 'Rating']])
    else:
        st.write('Silakan lengkapi semua input untuk melihat rekomendasi tempat wisata.')

# Tab kedua: Visualisasi Data
def visualisasi_data():
    viz_choice = st.radio("Pilih Visualisasi:", ("Tempat Wisata Terpopuler", "Perbandingan Kategori Wisata", "Distribusi Usia User", "Distribusi Harga Tiket Masuk", "Asal Kota Pengunjung"))

    if viz_choice == "Tempat Wisata Terpopuler":
        # Tempat wisata dengan jumlah rating terbanyak
        top_10 = rating['Place_Id'].value_counts().reset_index()[0:10]
        top_10 = pd.merge(top_10, place[['Place_Id', 'Place_Name']], how='left', left_on='Place_Id', right_on='Place_Id')
        plt.figure(figsize=(8, 5))
        sns.barplot(x='index', y='Place_Id', data=top_10)
        plt.title('Jumlah Tempat Wisata dengan Rating Terbanyak', pad=20)
        plt.ylabel('Jumlah Rating')
        plt.xlabel('Nama Lokasi')
        st.pyplot(plt)

    elif viz_choice == "Perbandingan Kategori Wisata":
        # Perbandingan jumlah kategori wisata
        plt.figure(figsize=(8, 5))
        sns.countplot(y='Category', data=place)
        plt.title('Perbandingan Jumlah Kategori Wisata', pad=20)
        st.pyplot(plt)

    elif viz_choice == "Distribusi Usia User":
        # Distribusi usia user
        plt.figure(figsize=(8, 5))
        sns.boxplot(user['Age'])
        plt.title('Distribusi Usia User', pad=20)
        st.pyplot(plt)

    elif viz_choice == "Distribusi Harga Tiket Masuk":
        # Distribusi harga masuk tempat wisata
        plt.figure(figsize=(8, 5))
        sns.boxplot(place['Price'])
        plt.title('Distribusi Harga Masuk Wisata', pad=20)
        st.pyplot(plt)

    elif viz_choice == "Asal Kota Pengunjung":
        # Visualisasi asal kota dari user
        askot = user['Location'].apply(lambda x: x.split(',')[0])
        plt.figure(figsize=(8, 6))
        sns.countplot(y=askot)
        plt.title('Jumlah Asal Kota dari User')
        st.pyplot(plt)

# Main App
st.title("Rekomendasi Tempat Wisata di Indonesia")

# Pilihan tab
tabs = ["Sistem Rekomendasi Wisata", "Visualisasi Data"]
choice = st.sidebar.radio("Pilihan Menu", tabs)

# Tampilkan tab yang dipilih
if choice == "Sistem Rekomendasi Wisata":
    filter_places()
elif choice == "Visualisasi Data":
    visualisasi_data()
