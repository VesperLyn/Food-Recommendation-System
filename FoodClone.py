import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from autocorrect import Speller
import random

model = SentenceTransformer('all-MiniLM-L6-v2')

# User profile
user_profile = st.sidebar
user_profile.write("Welcome, Mathew Johnson!")
user_image = "ProfilePic.jpg"
user_profile.image(user_image, width=100)
user_profile.write("Email: mathewjohnson@gmail.com")

st.write("## Food Recommendation System!")
st.image("./homepage.jpg", use_column_width=True)
st.write("Find your best food selection.")
st.write("Search by ingredient, cuisine, or dietary restriction to discover your new favorite dish.")

# input food ingredient or name
food_input = st.text_input("What food are you in the mood for?", key='food_input', placeholder="Enter food name / ingredient / description")
speller = Speller(lang='en')

# read data
df = pd.read_csv('./DatasetClone.csv')
df = df[['Food_Name', 'Food_Type', 'Food_Origin', 'Food_Description']]

def recommend_food(food_input, history):
    data = df[['Food_Name', 'Food_Type', 'Food_Origin', 'Food_Description']]

    # convert word to lowercase
    data['Food_Description'] = data['Food_Description'].str.lower()
    data['Food_Name'] = data['Food_Name'].str.lower()

    # concat food name and description
    data['Food_Text'] = data['Food_Name'] + ' ' + data['Food_Description']

    # converting to vectors using sbert
    vectors = model.encode(data['Food_Text'].tolist())

    user_vectors = model.encode([food_input.lower()])

    # finding similarities between vector
    similarities = cosine_similarity(user_vectors, vectors)

    # get 5 food with same similarity
    # ascending sort
    top_index = np.argsort(similarities[0])[-5:][::-1]
    recommendations = []
    history_weights = {}
    for word in history:
        if word in history_weights:
            history_weights[word] += 1
        else:
            history_weights[word] = 1

    weighted_similarities = similarities[0].copy()
    for i in range(len(weighted_similarities)):
        food_name = data['Food_Name'][i]
        food_description = data['Food_Description'][i]
        for word, freq in history_weights.items():
            if word in food_name or word in food_description:
                weighted_similarities[i] += freq * 0.1

    top_index = np.argsort(weighted_similarities)[-5:][::-1]
    for i in top_index:
        food_name = data['Food_Name'][i]
        food_description = data['Food_Description'][i]
        recommendations.append((food_name.capitalize(), food_description))
    return recommendations

if 'history' not in st.session_state:
    st.session_state.history = []

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = [(row['Food_Name'].capitalize(), row['Food_Description']) for index, row in df.sample(5).iterrows()]

if st.button('Recommend Food'):
    if food_input:
        st.session_state.history.append(food_input)
        st.session_state.recommendations = recommend_food(food_input, st.session_state.history)
        st.write("## Top 5 Food Recommendations")
    else:
        st.write("Please input a food name, ingredient, or description...")
else:
    st.write("## Today's Recommendations")

for i, (food_name, food_description) in enumerate(st.session_state.recommendations):
    with st.expander(food_name):
        st.write(f"{food_description}")

if 'history' in st.session_state:
    st.write("## History")
    for i, value in enumerate(st.session_state.history):
        st.write(f"{i+1}. {value}")