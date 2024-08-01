import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from autocorrect import Speller
import random

#Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

#User profile section
userProfile = st.sidebar
userProfile.write("Welcome, Mathew Johnson!")
userImage = "ProfilePic.jpg"
userProfile.image(userImage, width=100)
userProfile.write("Email: mathewjohnson@gmail.com")

#Homepage
st.write("## Food Recommendation System!")
st.image("./homepage.jpg", use_column_width=True)
st.write("Find your best food selection.")
st.write("Search by ingredient, cuisine, or dietary restriction to discover your new favorite dish.")

#Input keyword
foodInput = st.text_input("What food are you in the mood for?", key='foodInput', placeholder="Enter food name / ingredient / description")
speller = Speller(lang='en')

#Read dataset
dataset = pd.read_csv('./Food Dataset.csv')
dataset = dataset[['Food_Name', 'Food_Type', 'Food_Origin', 'Food_Description']]

def recommendFood(foodInput):
    #Make copy of dataset
    duplicate = dataset.copy()

    #Convert to lowercase
    duplicate['Food_Description'] = duplicate['Food_Description'].str.lower()
    duplicate['Food_Name'] = duplicate['Food_Name'].str.lower()

    #Encode food name and desc as individual
    vectors = model.encode(duplicate['Food_Description'].tolist() + duplicate['Food_Name'].tolist())

    userVectors = model.encode([foodInput])

    #Calculate similarity
    similarities = cosine_similarity(userVectors, vectors)

    #Display 5 recommendation with ascending sort
    topIndex = np.argsort(similarities[0])[-5:][::-1]
    recommendations = []
    for i in topIndex:
        if i < len(duplicate['Food_Name']):
            foodName = duplicate['Food_Name'][i]
            foodDescription = duplicate['Food_Description'][i]
        else:
            foodName = duplicate['Food_Name'][i - len(duplicate['Food_Name'])]
            foodDescription = duplicate['Food_Description'][i - len(duplicate['Food_Name'])]
        recommendations.append((foodName.capitalize(), foodDescription))
        if not recommendations:
            return "Sorry, we cannot find the food to recommend."
    return recommendations

if 'history' not in st.session_state:
    st.session_state.history = []

#Displaying today recommendation if history not exist
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = [(row['Food_Name'].capitalize(), row['Food_Description']) for index, row in dataset.sample(5).iterrows()]

#Recommend button
if st.button('Get Recommendation Now'):
    if foodInput:
        st.session_state.history.append(foodInput)
        st.session_state.recommendations = recommendFood(foodInput)
        st.write("## Top 5 Food Recommendations")
    else:
        st.write("Please input a food name, ingredient, or description...")
else:
    st.write("## Today's Recommendations")

#Display food name with drop down desc
for i, (foodName, foodDescription) in enumerate(st.session_state.recommendations):
    with st.expander(foodName):
        st.write(f"{foodDescription}")

#Display history
if 'history' in st.session_state:
    st.write("## History")
    for i, value in enumerate(st.session_state.history):
        st.write(f"{i+1}. {value}")