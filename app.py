import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load datasets
rating = pd.read_csv('tourism_rating.csv')
place = pd.read_csv('tourism_with_id.csv')
user = pd.read_csv('user.csv')

# Preprocessing and model training (same code as before)

# Define the RecommenderNet class and compile the model (same code as before)

# Define the Streamlit app
def main():
    st.title("Wisataku: Sistem Rekomendasi Tempat Wisata")

    # Sidebar navigation
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Rekomendasi"])

    if page == "Beranda":
        st.subheader("Selamat datang di Wisataku!")
        st.write("Nikmati liburan Anda dengan rekomendasi tempat wisata yang tepat.")

        # You can add more content here for the home page if needed

    elif page == "Rekomendasi":
        st.subheader("Rekomendasi Tempat Wisata")

        user_input = st.text_input("Masukkan ID Pengguna (User_ID):")
        if user_input:
            try:
                user_id = int(user_input)
                if user_id in user['User_Id'].values:
                    show_recommendations(user_id)
                else:
                    st.error("ID Pengguna tidak valid. Silakan masukkan ID Pengguna yang benar.")
            except ValueError:
                st.error("ID Pengguna harus berupa bilangan bulat.")
    
# Function to display recommendations
def show_recommendations(user_id):
    # Prepare data for recommendation (same code as before)
    # ...

    # Generate recommendations
    # ...

    # Display recommendations
    # ...

# Run the Streamlit app
if __name__ == "__main__":
    main()
