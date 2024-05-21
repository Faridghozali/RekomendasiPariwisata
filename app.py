import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache
def load_data():
    rating = pd.read_csv('tourism_rating.csv')
    place = pd.read_csv('tourism_with_id.csv')
    user = pd.read_csv('user.csv')
    return rating, place, user

rating, place, user = load_data()

# Drop unnecessary columns
place = place.drop(['Unnamed: 11', 'Unnamed: 12'], axis=1)
place = place.drop('Time_Minutes', axis=1)

# Filter ratings for places 
rating = pd.merge(rating, place[['Place_Id']], how='right', on='Place_Id')

# Filter users who have visited places
user = pd.merge(user, rating[['User_Id']], how='right', on='User_Id').drop_duplicates().sort_values('User_Id')

# Encoding function
def dict_encoder(col, data):
    unique_val = data[col].unique().tolist()
    val_to_val_encoded = {x: i for i, x in enumerate(unique_val)}
    val_encoded_to_val = {i: x for i, x in enumerate(unique_val)}
    return val_to_val_encoded, val_encoded_to_val

# Encoding User_Id and Place_Id
user_to_user_encoded, user_encoded_to_user = dict_encoder('User_Id', rating)
place_to_place_encoded, place_encoded_to_place = dict_encoder('Place_Id', rating)

rating['user'] = rating['User_Id'].map(user_to_user_encoded)
rating['place'] = rating['Place_Id'].map(place_to_place_encoded)

num_users, num_place = len(user_to_user_encoded), len(place_to_place_encoded)
rating['Place_Ratings'] = rating['Place_Ratings'].values.astype(np.float32)
min_rating, max_rating = min(rating['Place_Ratings']), max(rating['Place_Ratings'])

# Shuffle the dataset
df = rating.sample(frac=1, random_state=42)

# Prepare training and validation data
x = df[['user', 'place']].values
y = df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:]

# Define the RecommenderNet model
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_places, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_places = num_places
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.user_bias = layers.Embedding(num_users, 1)
        self.places_embedding = layers.Embedding(num_places, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
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

# Compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004), metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train the model
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_root_mean_squared_error') < 0.25:
            print('Lapor! Metriks validasi sudah sesuai harapan')
            self.model.stop_training = True

history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[myCallback()])

# Streamlit UI
st.sidebar.title("Pilih Sistem atau Visualisasi")
visualization_choice = st.sidebar.radio("Pilih opsi:", ("Sistem Rekomendasi Wisata", "Visualisasi Data"))

if visualization_choice == "Sistem Rekomendasi Wisata":
    st.title("Rekomendasi Pariwisata di Indonesia")

    user['Location'] = user['Location'].apply(lambda x: x.split(',')[0])
    user.columns = ['User_Id', 'Age', 'Location']

    # Pilihan input untuk Lokasi dan Umur
    user_location = st.selectbox("Pilih Lokasi", user['Location'].unique())
    user_age = st.slider("Pilih Umur", min_value=int(user['Age'].min()), max_value=int(user['Age'].max()), value=int(user['Age'].min()))

    # Mendapatkan User_Id yang sesuai dengan lokasi dan umur
    user_id = user[(user['Location'] == user_location) & (user['Age'] == user_age)]['User_Id'].values
    if len(user_id) > 0:
        user_id = user_id[0]
    else:
        st.write("User dengan lokasi dan umur tersebut tidak ditemukan.")
        st.stop()

    place_df = place[['Place_Id', 'Place_Name', 'Category', 'Rating', 'Price']]
    place_df.columns = ['id', 'place_name', 'category', 'rating', 'price']
    place_visited_by_user = df[df.User_Id == user_id]
    place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.Place_Id.values)]['id']
    place_not_visited = list(set(place_not_visited).intersection(set(place_to_place_encoded.keys())))
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_encoder = user_to_user_encoded.get(user_id)
    user_place_array = np.hstack(([[user_encoder]] * len(place_not_visited), place_not_visited))

    # Predict top 7 recommendations
    ratings = model.predict(user_place_array).flatten()
    top_ratings_indices = ratings.argsort()[-7:][::-1]
    recommended_place_ids = [place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices]

    st.write(f"Daftar rekomendasi untuk: Lokasi {user_location} dan Umur {user_age}")
    st.write("===" * 15)
    st.write("----" * 15)
    st.write("Tempat dengan rating wisata paling tinggi dari user")
    st.write("----" * 15)

    top_place_user = place_visited_by_user.sort_values(by='Place_Ratings', ascending=False).head(5).Place_Id.values
    place_df_rows = place_df[place_df['id'].isin(top_place_user)]
    for row in place_df_rows.itertuples():
        st.write(f"{row.place_name} : {row.category}")

    st.write("----" * 15)
    st.write("Top 7 place recommendation")
    st.write("----" * 15)

    recommended_place = place_df[place_df['id'].isin(recommended_place_ids)]
    for row, i in zip(recommended_place.itertuples(), range(1, 8)):
        st.write(f"{i}. {row.place_name}\n    {row.category}, Harga Tiket Masuk {row.price}, Rating Wisata {row.rating}\n")

    st.write("===" * 15)

elif visualization_choice == "Visualisasi Data":
    st.sidebar.title("Visualisasi Data")
    viz_choice = st.sidebar.radio("Pilih Visualisasi:", ("Tempat Wisata Terpopuler", "Perbandingan Kategori Wisata", "Distribusi Usia User", "Distribusi Harga Tiket Masuk", "Asal Kota Pengunjung"))

    if viz_choice == "Tempat Wisata Terpopuler":
        # Tempat wisata dengan jumlah rating terbanyak
        top_10 = rating['Place_Id'].value_counts().reset_index()[0:10]
        top_10 = pd.merge(top_10, place[['Place_Id', 'Place_Name']], how='left', left_on='Place_Id', right_on='Place_Id')
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Place_Id', y='Place_Name', data=top_10)
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
