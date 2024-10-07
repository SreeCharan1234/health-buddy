import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

# Set up the geolocator
geolocator = Nominatim(user_agent="geoapiExercises")

# Helper function to find nearby hospitals
def get_nearby_hospitals(location, radius_km=5):
    # Dummy data - In a real app, you would integrate with a hospital API or dataset
    hospitals = [
        {"name": "City Hospital", "location": (location[0] + 0.02, location[1] + 0.02)},
        {"name": "Health Clinic", "location": (location[0] - 0.01, location[1] + 0.03)},
        {"name": "MediCare Center", "location": (location[0] + 0.015, location[1] - 0.01)}
    ]
    
    # Filter hospitals by distance
    nearby_hospitals = []
    for hospital in hospitals:
        distance = geodesic(location, hospital['location']).km
        if distance <= radius_km:
            hospital['distance'] = round(distance, 2)
            nearby_hospitals.append(hospital)
    
    return nearby_hospitals

# Streamlit interface
st.title("Nearby Hospitals Finder")
location_input = st.text_input("Enter your location (e.g., city, landmark):", "New York")

if location_input:
    # Get the coordinates of the entered location
    location = geolocator.geocode(location_input)
    
    if location:
        st.write(f"Location: {location.address}")
        
        # Get nearby hospitals
        nearby_hospitals = get_nearby_hospitals((location.latitude, location.longitude))
        
        # Display map
        map_center = [location.latitude, location.longitude]
        m = folium.Map(location=map_center, zoom_start=14)
        
        # Add marker for user's location
        folium.Marker(map_center, tooltip="Your Location", icon=folium.Icon(color="blue")).add_to(m)
        
        # Add hospitals to the map
        for hospital in nearby_hospitals:
            folium.Marker(
                hospital['location'],
                tooltip=f"{hospital['name']} ({hospital['distance']} km)",
                icon=folium.Icon(color="red")
            ).add_to(m)
        
        st_folium(m, width=700, height=500)
        
        # List hospitals in Streamlit
        st.write("### Nearby Hospitals")
        for hospital in nearby_hospitals:
            st.write(f"🏥 {hospital['name']} - {hospital['distance']} km away")
    else:
        st.error("Location not found. Please try again.")




"""
import streamlit as st
from gtts import gTTS
import os

# Function to convert text to speech
def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language)
    tts.save("output.mp3")
    return "output.mp3"

# Streamlit UI
st.title("Text to Speech Converter")

# Input for text
text_input = st.text_area("Enter text here:")

# Dropdown to select language
language_option = st.selectbox("Select Language", 
                               ("English", "Hindi", "Telugu"))

# Map language option to gTTS language codes
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te"
}

if st.button("Convert to Speech"):
    if text_input:
        selected_language = language_map[language_option]
        output_file = text_to_speech(text_input, selected_language)
        
        audio_file = open(output_file, 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='audio/mp3')
    else:
        st.error("Please enter some text.")


"""