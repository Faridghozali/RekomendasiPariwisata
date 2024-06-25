import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load datasets
rating = pd.read_csv('tourism_rating.csv')
place = pd.read_csv('tourism_with_id.csv')
user = pd.read_csv('user.csv')

# Data cleaning
place = place.drop(['Unnamed: 11','Unnamed: 12', 'Time_Minutes'], axis=1)

# Filter ratings for places in Surabaya
rating = pd.merge(rating, place[['Place_Id']], how='right', on='Place_Id')

# Filter users who have visited at least one place
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

# Shuffle and split data
df = rating.sample(frac=1, random_state=42)
x = df[['user', 'place']].values
y = df['Place_Ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

# Recommender model
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_places, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_places = num_places
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users, embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.places_embedding = layers.Embedding(
            num_places, embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
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

model = RecommenderNet(num_users, num_place, 50)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_root_mean_squared_error') < 0.25:
            self.model.stop_training = True

# Train the model
history = model.fit(
    x=x_train,
    y=y_train,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=[myCallback()]
)

# Streamlit app
st.title('Rekomendasi Pariwisata')

user_id = st.number_input('Masukkan User ID', min_value=0, max_value=num_users - 1, value=0, step=1)
if st.button('Dapatkan Rekomendasi'):
    place_df = place[['Place_Id', 'Place_Name', 'Category', 'Rating', 'Price']]
    place_df.columns = ['id', 'place_name', 'category', 'rating', 'price']

    place_visited_by_user = df[df.User_Id == user_id]
    place_not_visited = place_df[~place_df['id'].isin(place_visited_by_user.Place_Id.values)]['id']
    place_not_visited = list(set(place_not_visited).intersection(set(place_to_place_encoded.keys())))
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_encoder = user_to_user_encoded.get(user_id)
    user_place_array = np.hstack(([[user_encoder]] * len(place_not_visited), place_not_visited))

    ratings = model.predict(user_place_array).flatten()
    top_ratings_indices = ratings.argsort()[-7:][::-1]
    recommended_place_ids = [place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices]

    st.write('Daftar rekomendasi untuk: User', user_id)
    st.write('Tempat dengan rating wisata paling tinggi dari user:')
    top_place_user = (
        place_visited_by_user.sort_values(by='Place_Ratings', ascending=False).head(5).Place_Id.values
    )
    place_df_rows = place_df[place_df['id'].isin(top_place_user)]
    for row in place_df_rows.itertuples():
        st.write(row.place_name, ':', row.category)

    st.write('Top 7 place recommendation:')
    recommended_place = place_df[place_df['id'].isin(recommended_place_ids)]
    for row, i in zip(recommended_place.itertuples(), range(1, 8)):
        st.write(f"{i}. {row.place_name} \n    {row.category}, Harga Tiket Masuk {row.price}, Rating Wisata {row.rating}")
