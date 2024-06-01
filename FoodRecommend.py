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
df = pd.read_csv('./Food Dataset.csv')
df = df[['Food_Name', 'Food_Type', 'Food_Origin', 'Food_Description']]

def recommend_food(food_input):
    data = df[['Food_Name', 'Food_Type', 'Food_Origin', 'Food_Description']]

    # convert word to lowercase
    data['Food_Description'] = data['Food_Description'].str.lower()
    data['Food_Name'] = data['Food_Name'].str.lower()

    # converting to vectors using sbert
    vectors = model.encode(data['Food_Description'].tolist() + data['Food_Name'].tolist())

    user_vectors = model.encode([food_input])

    # finding similarities between vector
    similarities = cosine_similarity(user_vectors, vectors)

    # get 5 food with same similarity
    # ascending sort
    top_indices = np.argsort(similarities[0])[-5:][::-1]
    recommendations = []
    for i in top_indices:
        if i < len(data['Food_Name']):
            food_name = data['Food_Name'][i]
            food_description = data['Food_Description'][i]
        else:
            food_name = data['Food_Name'][i - len(data['Food_Name'])]
            food_description = data['Food_Description'][i - len(data['Food_Name'])]
        recommendations.append((food_name.capitalize(), food_description))
        if not recommendations:
            return "Sorry, we cannot find the food to recommend."
    return recommendations

if 'history' not in st.session_state:
    st.session_state.history = []

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = [(row['Food_Name'].capitalize(), row['Food_Description']) for index, row in df.sample(5).iterrows()]

if st.button('Recommend Food'):
    if food_input:
        st.session_state.history.append(food_input)
        st.session_state.recommendations = recommend_food(food_input)
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