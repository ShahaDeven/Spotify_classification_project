import streamlit as st
import librosa
import numpy as np
import requests
import os
from pydub import AudioSegment
import uuid


# Define FastAPI URL for prediction (local or cloud)
FASTAPI_URL = "https://music-classifier-backend-nqly6c27da-uc.a.run.app/predict/"  # Update for local or cloud deployment

st.title("MP3 Song Mood Classifier & Metadata Extractor")
uploaded_file = st.file_uploader("Upload a song file (MP3 format)", type=["mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")
    st.write("### Processing Audio File...")

    file_path = f"{uuid.uuid4()}.mp3"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        files = {"file": (file_path, open(file_path, "rb"), "audio/mpeg")}
        response = requests.post(FASTAPI_URL, files=files)
        
        if response.status_code == 200:
            prediction = response.json()
            mood = prediction.get("mood", "Unknown")
            metadata = prediction.get("song_metadata", {})
    
            st.write("### Song Metadata:")
            st.write(f"**Title:** {metadata.get('Title', 'Unknown')}")
            st.write(f"**Artist:** {metadata.get('Artist', 'Unknown')}")
            st.write(f"**Album:** {metadata.get('Album', 'Unknown')}")
            st.write(f"**Release Date:** {metadata.get('Release Date', 'Unknown')}")
            st.write(f"### Predicted Mood: {mood}")

        else:
            st.error(f"Error fetching prediction from FastAPI. Status: {response.status_code}")
            st.write(response.text)
    except Exception as e:
        st.error(f"Error communicating with FastAPI: {e}")
