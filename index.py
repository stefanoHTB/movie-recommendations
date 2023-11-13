import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Concatenate

# Sample data: User IDs, Movie IDs, and Ratings
user_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3])
movie_ids = np.array([0, 0, 0, 0, 1, 1, 1, 1])
ratings = np.array([5, 4, 3, 2, 4, 3, 5, 1])

# Number of users and movies
num_users = len(np.unique(user_ids))
num_movies = len(np.unique(movie_ids))

# Embedding size (you can experiment with different values)
embedding_size = 10

# User embedding layer
user_input = Input(shape=(1,), name='user_input')
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, input_length=1)(user_input)
user_embedding = Flatten()(user_embedding)

# Movie embedding layer
movie_input = Input(shape=(1,), name='movie_input')
movie_embedding = Embedding(input_dim=num_movies, output_dim=embedding_size, input_length=1)(movie_input)
movie_embedding = Flatten()(movie_embedding)

# Dot product to get user-movie interactions
dot_product = Dot(axes=1)([user_embedding, movie_embedding])

# Combine user and movie embeddings into a single model
model = Model(inputs=[user_input, movie_input], outputs=dot_product)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit([user_ids, movie_ids], ratings, epochs=50, verbose=1)

# Make predictions
user_ids_to_predict = np.array([0, 1, 2, 3])
movie_ids_to_predict = np.array([2, 2, 2, 2])

predictions = model.predict([user_ids_to_predict, movie_ids_to_predict])

# Display predictions
for i in range(len(user_ids_to_predict)):
    print(f"User {user_ids_to_predict[i]} rating for Movie {movie_ids_to_predict[i]}: {predictions[i][0]:.2f}")
