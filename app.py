import streamlit as st
import pandas as pd
import numpy as np
from zipfile import ZipFile
from pathlib import Path
import seaborn as sns
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
    sns.barplot(x='Place_Id', y='Place_Name', data=top_10)
    plt.title('Jumlah Tempat Wisata dengan Rating Terbanyak', pad=20)
    plt.ylabel('Jumlah Rating')
    plt.xlabel('Nama Lokasi')
    st.pyplot(plt.gcf())
    
    sns.countplot(y='Category', data=place)
    plt.title('Perbandingan Jumlah Kategori Wisata ', pad=20)
    st.pyplot(plt.gcf())
    
    plt.figure(figsize=(5,3))
    sns.boxplot(user['Age'])
    plt.title('Distribusi Usia User', pad=20)
    st.pyplot(plt.gcf())
    
    plt.figure(figsize=(7,3))
    sns.boxplot(place['Price'])
    plt.title('Distribusi Harga Masuk Wisata ', pad=20)
    st.pyplot(plt.gcf())
    
    askot = user['Location'].apply(lambda x : x.split(',')[0])
    
    plt.figure(figsize=(8,6))
    sns.countplot(y=askot)
    plt.title('Jumlah Asal Kota dari User')
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
        optimizer=keras.optimizers.Adam(learning_rate=0.0004),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('val_root_mean_squared_error') < 0.25):
                st.write('Lapor! Metriks validasi sudah sesuai harapan')
                self.model.stop_training = True
    
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=100,
        validation_data=(x_val, y_val),
        callbacks=[myCallback()]
    )
    
    st.subheader("Plot Loss dan Validation")
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('model_metrics')
    plt.ylabel('root_mean_squared_error')
    plt.xlabel('epoch')
    plt.ylim(ymin=0, ymax=0.4)
    plt.legend(['train', 'test'], loc='center left')
    st.pyplot(plt.gcf())
    
    place_df = place[['Place_Id','Place_Name','Category','Rating','Price']]
    place_df.columns = ['id','place_name','category','rating','price']
    df = rating.copy()
    
    user_id = st.selectbox('Pilih User ID', df.User_Id.unique())
    place_visited_by_user = df[df.User_Id == user_id]
    
    place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.Place_Id.values)]['id']
    place_not_visited = list(set(place_not_visited).intersection(set(place_to_place_encoded.keys())))
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_encoder = user_to_user_encoded.get(user_id)
    user_place_array = np.hstack(([[user_encoder]] * len(place_not_visited), place_not_visited))
    
    ratings = model.predict(user_place_array).flatten()
    top_ratings_indices = ratings.argsort()[-7:][::-1]
    recommended_place_ids = [place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices]
    
    st.subheader('Daftar Rekomendasi')
    st.write(f"Rekomendasi untuk User {user_id}")
    
    top_place_user = place_visited_by_user.sort_values(by='Place_Ratings', ascending=False).head(5).Place_Id.values
    place_df_rows = place_df[place_df['id'].isin(top_place_user)]
    
    st.write('Tempat dengan rating wisata paling tinggi dari user:')
    for row in place_df_rows.itertuples():
        st.write(f"{row.place_name} : {row.category}")
    
    st.write('Top 7 rekomendasi tempat wisata:')
    recommended_place = place_df[place_df['id'].isin(recommended_place_ids)]
    for row, i in zip(recommended_place.itertuples(), range(1, 8)):
        st.write(f"{i}. {row.place_name} \n    {row.category}, Harga Tiket Masuk {row.price}, Rating Wisata {row.rating}")

if __name__ == "__main__":
    main()

