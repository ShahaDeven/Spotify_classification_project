import os
from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
import librosa
import numpy as np
import requests
import time
import hmac
import hashlib
import base64
from pydub import AudioSegment
import uvicorn
import uuid
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Load the pre-trained model
model = joblib.load("random_forest_model.pkl")

# ACRCloud API details (replace with your actual credentials)
access_key = "b64497762cda27a0361fd5a4a75a2e74"
access_secret = "WFbyRGnZ9e0zlbnsoReqEGLscIJ5XosuDfaSZpY6"
requrl = "https://identify-us-west-2.acrcloud.com/v1/identify"

FEATURE_COLUMNS = [
    "danceability", "energy", "normalized_loudness", "tempo", "valence", "instrumentalness", "liveness"
]

@app.post("/predict/")
async def predict_mood(file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename}")
    try:
        unique_file_id = str(uuid.uuid4())
        file_path = f"temp_{unique_file_id}.mp3"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        try:
            song = AudioSegment.from_file(file_path)
            duration_ms = len(song)
            start_time = 0
            end_time = min(10 * 1000, duration_ms)  
            clipped_audio = song[start_time:end_time]
            clipped_path = "temp_clipped_audio.mp3"
            clipped_audio.export(clipped_path, format="mp3")
            print("Extracted a 10-second audio clip for ACRCloud.")
        except Exception as e:
            os.remove(file_path)
            raise Exception(f"Error processing audio file: {e}")

        try:
            sample_bytes = os.path.getsize(clipped_path)
            with open(clipped_path, "rb") as f:
                audio_data = f.read()

            timestamp = int(time.time())
            string_to_sign = f"POST\n/v1/identify\n{access_key}\naudio\n1\n{timestamp}"
            sign = base64.b64encode(
                hmac.new(access_secret.encode("ascii"), string_to_sign.encode("ascii"), digestmod=hashlib.sha1).digest()
            ).decode("ascii")

            headers = {}
            data = {
                "access_key": access_key,
                "sample_bytes": sample_bytes,
                "timestamp": str(timestamp),
                "signature": sign,
                "data_type": "audio",
                "signature_version": "1",
            }
            files = {"sample": ("audio.mp3", audio_data, "audio/mpeg")}
            response = requests.post(requrl, data=data, files=files, headers=headers)

            acr_data = response.json()

            if "status" in acr_data and acr_data["status"]["code"] == 0:
                metadata = acr_data.get("metadata", {}).get("music", [{}])[0]
                song_metadata = {
                    "Title": metadata.get("title", "Unknown"),
                    "Artist": metadata.get("artists", [{}])[0].get("name", "Unknown"),
                    "Album": metadata.get("album", {}).get("name", "Unknown"),
                    "Release Date": metadata.get("release_date", "Unknown"),
                }
            else:
                song_metadata = {"error": f"ACRCloud API Error: {acr_data['status'].get('msg', 'Unknown error')}"}
        except Exception as e:
            song_metadata = {"error": f"Error querying ACRCloud API: {e}"}

        y, sr = librosa.load(file_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        loudness = 10 * np.log10(np.mean(y**2))
        normalized_loudness = (loudness + 50) / 50  
        danceability = librosa.feature.tempogram(y=y, sr=sr).mean()
        valence = normalized_loudness
        energy = np.mean(librosa.feature.rms(y=y))
        instrumentalness = np.mean(np.abs(librosa.effects.hpss(y)[0])) / np.mean(np.abs(y))
        liveness = np.mean(librosa.feature.spectral_flatness(y=y))

        features = pd.DataFrame([{
            "danceability": danceability,
            "energy": energy,
            "normalized_loudness": normalized_loudness,
            "tempo": tempo,
            "valence": valence,
            "instrumentalness": instrumentalness,
            "liveness": liveness
        }])

        features = features[FEATURE_COLUMNS]

        prediction = model.predict(features)[0]
        label_to_mood = {
            0: "Calm", 1: "Energetic", 2: "Happy", 
            3: "Instrumental", 4: "Live", 5: "Neutral", 6: "Sad"
        }
        mood = label_to_mood.get(prediction, "Unknown")

        return {"mood": mood}
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {"error": f"Failed to process the file: {str(e)}"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)