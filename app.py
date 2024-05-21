import streamlit as st
import pandas as pd
import numpy as np
from zipfile import ZipFile
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
place_df = pd.read_csv("https://raw.githubusercontent.com/khikisb/SistemRekomendasiWisata/main/tourism_with_id.csv")
rating = pd.read_csv('https://raw.githubusercontent.com/khikisb/SistemRekomendasiWisata/main/tourism_rating.csv')
user = pd.read_csv('https://raw.githubusercontent.com/khikisb/SistemRekomendasiWisata/main/user.csv')

# Data preprocessing
place_df = place_df.drop(['Unnamed: 11','Unnamed: 12', 'Time_Minutes'], axis=1)
rating = pd.merge(rating, place_df[['Place_Id']], how='right', on='Place_Id')
user = pd.merge(user, rating[['User_Id']], how='right', on='User_Id').drop_duplicates().sort_values('User_Id')

# Data encoding
def dict_encoder(col, data):
    unique_val = data[col].unique().tolist()
    val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
    return val_to_val_encoded, val_encoded_to_val

user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id', user)
place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id', place_df)

rating['user'] = rating['User_Id'].map(user_to_user_encoded)
rating['place'] = rating['Place_Id'].map(place_to_place_encoded)

num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)

min_rating, max_rating = min(rating['Place_Ratings']), max(rating['Place_Ratings'])

# Model definition
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_places, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_places = num_places
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.places_embedding = layers.Embedding(
            num_places,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.places_bias = layers.Embedding(num_places, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        places_vector = self.places_embedding(inputs[:, 1])
        places_bias = self.places_bias(inputs[:, 1])

        dot_user_places = tf.tensordot(user_vector, places_vector, 2)

        x = dot_user_places + user_bias + places_bias

        return tf.nn.sigmoid(x)

# Load the trained model
model = RecommenderNet(num_users, num_place, 50)
model.load_weights("model_weights.h5")

# Streamlit App
st.title("Sistem Rekomendasi Tempat Wisata")

tabs = ["Filter Tempat Wisata", "Rekomendasi berdasarkan Deskripsi"]
choice = st.sidebar.radio("Navigasi", tabs)

if choice == "Filter Tempat Wisata":
    st.sidebar.title('Filter Tempat Wisata')
    min_price = place_df['Price'].min()
    max_price = place_df['Price'].max()
    categories = st.sidebar.selectbox('Category wisata?', place_df['Category'].unique())
    cities = st.sidebar.selectbox('Lokasi?', place_df['City'].unique())
    selected_price_range = st.sidebar.slider('Range Harga?', min_value=min_price, max_value=max_price, value=(min_price, max_price))

    min_price, max_price = selected_price_range

    filtered_data = place_df[(place_df['Category'] == categories) &
                             (place_df['City'] == cities) &
                             (place_df['Price'] >= min_price) &
                             (place_df['Price'] <= max_price)]

    st.header('Tempat Wisata yang Sesuai dengan Preferensi Kamu')
    if len(filtered_data) == 0:
        st.write('Maaf, tidak ada tempat wisata yang sesuai dengan preferensi Kamu.')
    else:
        st.write(filtered_data[['Place_Name', 'Description', 'Category', 'City', 'Price', 'Rating']])

elif choice == "Rekomendasi berdasarkan Deskripsi":
    user_input = st.text_area("Ceritakan kamu mau pergi kemana? dengan siapa?dan ingin melakukan apa?")
    st.write('Contoh : saya ingin pergi dengan keluarga dan ingin melihat lukisan lukisan yang indah')
    st.write('Contoh : saya ingin pergi ke pantai yang masih jarang orang tahu')
    if user_input:
        user_tfidf = tfidf.transform([user_input])
        similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)
        recommended_indices = similarity_scores.argsort()[0][::-1][:5]
        recommended_places = place_df.iloc[recommended_indices][['Place_Name', 'Description', 'Category', 'City', 'Price', 'Rating']]
        st.write("Tempat wisata yang direkomendasikan berdasarkan deskripsi Kamu:")
        st.write(recommended_places)
    else:
        st.write("Hindari menggunakan nama kota, Karena kami akan merekomendasikan tempat yang paling cocok dengan Kamu di Seluruh Indonesia")
