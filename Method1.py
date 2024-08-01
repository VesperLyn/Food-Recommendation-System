import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from autocorrect import Speller

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
dataset = pd.read_csv('./DatasetClone.csv')
dataset = dataset[['Food_Name', 'Food_Type', 'Food_Origin', 'Food_Description']]

def recommendFood(foodInput, history):
    #Encode food name and desc together
    foodVectors = model.encode([foodInput.lower()])
    datasetVectors = model.encode(dataset['Food_Name'] + ' ' + dataset['Food_Description'])

    #Calculate similarity
    similarities = cosine_similarity(foodVectors, datasetVectors)[0]

    #Add weight(value) to each history
    historyWeights = {}
    for word in history:
        for i, row in dataset.iterrows():
            if word in row['Food_Name'] or word in row['Food_Description']:
                historyWeights[i] = historyWeights.get(i, 0) + 0.1

    #Add history weights to similarities
    for i, similarity in enumerate(similarities):
        similarities[i] += historyWeights.get(i, 0)

    #Get top 5 recommendations
    topIndex = np.argsort(similarities)[-5:][::-1]
    recommendations = [(dataset['Food_Name'][i].capitalize(), dataset['Food_Description'][i], similarities[i]) for i in topIndex]
    return recommendations

if 'history' not in st.session_state:
    st.session_state.history = []

#Displaying today recommendation if history not exist
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = [(row['Food_Name'].capitalize(), row['Food_Description'], 0) for index, row in dataset.sample(5).iterrows()]

#Recommend button
if st.button('Get Recommendation Now'):
    if foodInput:
        st.session_state.history.append(foodInput)
        st.session_state.recommendations = recommendFood(foodInput, st.session_state.history)
        st.write("## Top 5 Food Recommendations")
    else:
        st.write("Please input a food name, ingredient, or description...")
else:
    st.write("## Today's Recommendations")

#Display food name with drop down desc and similarity score
for i, (foodName, foodDescription, similarity) in enumerate(st.session_state.recommendations):
    with st.expander(foodName):
        st.write(f"{foodDescription} (Similarity: {similarity:.2f})")

#Display history
if 'history' in st.session_state:
    st.write("## History")
    for i, value in enumerate(st.session_state.history):
        st.write(f"{i+1}. {value}")