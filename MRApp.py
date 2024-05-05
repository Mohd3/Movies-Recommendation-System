import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# Load the Universal Sentence Encoder model
model_url = 'https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2'
model = hub.load(model_url)

# Load the movies dataset
df = pd.read_csv('movies.csv')

# Clean the dataset
df = df[['title', 'overview']].dropna()

overviews = list(df['overview'])

embeddings = model(overviews)

# Fit nearest neighbors model
nn = NearestNeighbors(n_neighbors=10)
nn.fit(embeddings)

# Recommender function
# Recommender function
def recommender(text):
    emb = model([text])
    neighbors = nn.kneighbors(emb, return_distance=False)[0]
    # Exclude the input text itself from the recommendations
    neighbors = [neighbor for neighbor in neighbors if df.iloc[neighbor]['title'].lower() != text.lower()]
    return df.iloc[neighbors]


# Streamlit interface
st.title("🍿 Movie Recommender")

# Input text box for user input
text = st.text_input("Enter a genre or your favourite movie: ")

# Call the recommender function when the user clicks the button
if st.button("Get Recommendations"):
    # Get recommendations based on user input
    recommendations = recommender(text)
    
    # Display recommendations
    st.write("Recommended Movies:")
    st.table(recommendations[['title', 'overview']].reset_index(drop=True))  # Reset index to remove it
