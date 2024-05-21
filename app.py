import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Fungsi untuk mengunggah file
def upload_file():
    uploaded_files = st.file_uploader("Upload dataset files", accept_multiple_files=True)
    file_dict = {}
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        file_dict[uploaded_file.name] = df
    return file_dict

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Rekomendasi Pariwisata")
    st.write("Aplikasi ini memberikan rekomendasi tempat wisata berdasarkan data yang ada.")
    
    # Unggah file dataset
    file_dict = upload_file()
    
    if len(file_dict) < 3:
        st.warning("Silakan unggah ketiga file dataset: 'tourism_rating.csv', 'tourism_with_id.csv', dan 'user.csv'.")
        return
    
    rating = file_dict['tourism_rating.csv']
    place = file_dict['tourism_with_id.csv']
    user = file_dict['user.csv']
    
    # Pratinjau data
    st.subheader("Pratinjau Data")
    st.write("Data Tempat Wisata:")
    st.dataframe(place.head())
    st.write("Data Rating:")
    st.dataframe(rating.head())
    st.write("Data User:")
    st.dataframe(user.head())
    
    # Preprocessing data
    place = place.drop(['Unnamed: 11','Unnamed: 12'],axis=1)
    place = place.drop('Time_Minutes', axis=1)
    
    rating = pd.merge(rating, place[['Place_Id']], how='right', on='Place_Id')
    
    user = pd.merge(user, rating[['User_Id']], how='right', on='User_Id').drop_duplicates().sort_values('User_Id')
    
    # Visualisasi
    st.subheader("Visualisasi Data")
    
    top_10 = rating['Place_Id'].value_counts().reset_index()[0:10]
    top_10 = pd.merge(top_10, place[['Place_Id','Place_Name']], how='left', left_on='Place_Id', right_on='Place_Id')
    
    plt.figure(figsize=(8,5))
    plt.bar(top_10['Place_Id'], top_10['Place_Name'])
    plt.title('Jumlah Tempat Wisata dengan Rating Terbanyak', pad=20)
    plt.ylabel('Jumlah Rating')
    plt.xlabel('Nama Lokasi')
    st.pyplot(plt.gcf())
    
    plt.figure(figsize=(6,4))
    place['Category'].value_counts().plot(kind='bar')
    plt.title('Perbandingan Jumlah Kategori Wisata ', pad=20)
    plt.xlabel('Kategori')
    plt.ylabel('Jumlah')
    st.pyplot(plt.gcf())
    
    plt.figure(figsize=(5,3))
    plt.boxplot(user['Age'])
    plt.title('Distribusi Usia User', pad=20)
    plt.xlabel('Usia')
    st.pyplot(plt.gcf())
    
    plt.figure(figsize=(5,3))
    plt.boxplot(place['Price'])
    plt.title('Distribusi Harga Masuk Wisata ', pad=20)
    plt.xlabel('Harga')
    st.pyplot(plt.gcf())
    
    askot = user['Location'].apply(lambda x : x.split(',')[0])
    
    plt.figure(figsize=(8,6))
    askot.value_counts().plot(kind='barh')
    plt.title('Jumlah Asal Kota dari User')
    plt.xlabel('Jumlah User')
    plt.ylabel('Kota')
    st.pyplot(plt.gcf())
    
    # Encoding data
    df = rating.copy()
    
    def dict_encoder(col, data=df):
        unique_val = data[col].unique().tolist()
        val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
        val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
        return val_to_val_encoded, val_encoded_to_val
    
    user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id')
    df['user'] = df['User_Id'].map(user_to_user_encoded)
    place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id')
    df['place'] = df['Place_Id'].map(place_to_place_encoded)
    
    num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)
    df['Place_Ratings'] = df['Place_Ratings'].values.astype(np.float32)
    
    min_rating, max_rating = min(df['Place_Ratings']), max(df['Place_Ratings'])
    
    st.write(f'Number of User: {num_users}, Number of Place: {num_place}, Min Rating: {min_rating}, Max Rating: {max_rating}')
    
    df = df.sample(frac=1, random_state=42)
    
    x = df[['user', 'place']].values
    y = df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    
    train_indices = int(0.8 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:]
    )
    
    class RecommenderNet(tf.keras.Model):
        def __init__(self, num_users, num_places, embedding_size, **kwargs):
            super(RecommenderNet, self).__init__(**kwargs)
            self.num_users = num_users
            self.num_places = num_places
            self.embedding_size = embedding_size
            self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
            self.user_bias = layers.Embedding(num_users, 1)
            self.places_embedding = layers.Embedding(num_places, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-6))
            self.places_bias = layers.Embedding(num_places, 1)
        
        def call(self, inputs):
            user_vector = self.user_embedding(inputs[:, 0])
            user_bias = self.user_bias(inputs[:, 0])
            places_vector = self.places_embedding(inputs[:, 1])
            places_bias = self.places_bias(inputs[:, 1])
            
            dot_user_places = tf.tensordot(user_vector, places_vector, 2)
            x = dot_user_places + user_bias + places_bias
            return tf.nn.sigmoid(x)
    
    model = RecommenderNet(num_users, num_place, 50)
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers
