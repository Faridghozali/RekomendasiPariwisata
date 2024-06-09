import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers

# Load dataset
def load_data():
    rating = pd.read_csv('tourism_rating.csv')
    place = pd.read_csv('tourism_with_id.csv')
    user = pd.read_csv('user.csv')
    return rating, place, user

# Drop unnecessary columns and perform necessary preprocessing
def preprocess_data(rating, place, user):
    # CSS for background images and custom styling
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://islamic-center.or.id/wp-content/uploads/2016/07/Pariwisata-Halal-Indonesia.jpg");
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
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Drop unnecessary columns
    place = place.drop(['Unnamed: 11', 'Unnamed: 12', 'Time_Minutes'], axis=1)

    # Filter ratings for places 
    rating = pd.merge(rating, place[['Place_Id']], how='right', on='Place_Id')

    # Shuffle the dataset
    df = rating.sample(frac=1, random_state=42)

    # Prepare training and validation data
    x = df[['user', 'place']].values
    y = df['Place_Ratings'].values.astype(np.float32)
    train_indices = int(0.8 * df.shape[0])
    x_train, x_val, y_train, y_val = x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:]

    return df, x_train, x_val, y_train, y_val

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

# Main App
def main():
    rating, place, user = load_data()
    df, x_train, x_val, y_train, y_val = preprocess_data(rating, place, user)

    # Define model
    num_users = len(np.unique(df['user']))
    num_places = len(np.unique(df['place']))
    model = RecommenderNet(num_users, num_places, 50)

    # Compile model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004), metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # Train model
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

    # Interface
    st.title("Rekomendasi Tempat Wisata di Indonesia")
    tabs = ["Sistem Rekomendasi Wisata", "Filter berdasarkan User", "Visualisasi Data"]
    choice = st.sidebar.radio("Pilihan Menu", tabs)

    if choice == "Sistem Rekomendasi Wisata":
        # Input user for name and age
        name = st.text_input('Masukkan nama kamu:')
        age = st.number_input('Masukkan umur kamu:', min_value=10, max_value=100)
        
        categories = st.selectbox('Kategori wisata', place['Category'].unique())
        cities = st.selectbox('Lokasi kamu', place['City'].unique())

        # Tampilkan hasil filter hanya jika semua inputan sudah terisi
        if name and age and categories and cities:
            # Filter data berdasarkan input pengguna
            filtered_data = place[(place['Category'] == categories) & (place['City'] == cities)]

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
            
    elif choice == "Filter berdasarkan User":
    user_id = st.selectbox("Pilih User ID", user['User_Id'].unique())
    
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
    
    st.write(f"Daftar rekomendasi untuk: User {user_id}")
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

elif choice == "Visualisasi Data":
    viz_choice = st.radio("Pilih Visualisasi:", ("Tempat Wisata Terpopuler", "Perbandingan Kategori Wisata", "Distribusi Usia User", "Distribusi Harga Tiket Masuk", "Asal Kota Pengunjung"))

    if viz_choice == "Tempat Wisata Terpopuler":
        # Tempat wisata dengan jumlah rating terbanyak
        top_10 = rating['Place_Id'].value_counts().reset_index().head(10)
        top_10 = pd.merge(top_10, place[['Place_Id', 'Place_Name']], how='left', left_on='Place_Id', right_on='Place_Id')
        top_10.columns = ['Place_Id', 'Jumlah_Rating', 'Place_Name']
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Place_Name', y='Jumlah_Rating', data=top_10)
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
if __name__ == "__main__":
    main()
