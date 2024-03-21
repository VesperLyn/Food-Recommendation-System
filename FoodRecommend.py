import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

if 'history' not in st.session_state:
    st.session_state.history = []

# User profile
user_profile = st.sidebar
user_profile.write("Welcome, Mathew Johnson!")
user_image = "ProfilePic.jpg"
user_profile.image(user_image, width=100)
user_profile.write("Email: mathewjohnson@gmail.com")

st.write("## Food Recommendation System!")
st.write("Find your best food selection!")

st.write("Search by ingredient, cuisine, or dietary restriction to discover your new favorite dish.")

# input food ingredient or name
food_input = st.text_input("What food are you in the mood for?", key='food_input')

# read data
df = pd.read_csv('./Food Dataset.csv')
df = df[['Customer', 'Food_Name', 'Food_Description']]

def recommend_food(food_input):
    data = df[['Food_Name', 'Food_Description']]

    # convert word to lowercase
    data['Food_Description'] = data['Food_Description'].str.lower()

    # converting to vectors using sbert
    vectors = model.encode(data['Food_Description'].tolist())

    user_vectors = model.encode([food_input])

    similarities = cosine_similarity(user_vectors, vectors)

    match_found = np.where(similarities >= 0.5)

if food_input:
    st.session_state.history.append(food_input)

if st.session_state.history:
    st.write("## History")
    for i, value in enumerate(st.session_state.history):
       st.write(f"{i+1}. {value}")