import streamlit as st

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

food_input = st.text_input("What food are you in the mood for?", key='food_input')

if food_input:
    st.session_state.history.append(food_input)

if st.session_state.history:
    st.write("## History")
    for i, value in enumerate(st.session_state.history):
        st.write(f"{i+1}. {value}")
