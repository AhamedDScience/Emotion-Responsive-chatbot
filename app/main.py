# Extra libraries
import os
import re
import cv2
import pickle
import numpy as np
import pandas as pd
import moviepy.editor as mp
from collections import Counter
from pyht import Client, TTSOptions, Format
from moviepy.video.io.VideoFileClip import VideoFileClip
from tensorflow import keras
from keras.models import model_from_json
import speech_recognition as sr
import face_recognition
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts.prompt import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

# Play HT TTS Client
client = Client("Client ID", "API Password")

# Google API Key setup
os.environ["GOOGLE_API_KEY"] = 'Google API Code'
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.6)

prompt_template = PromptTemplate(
    input_variables=['history', 'input'],
    template='''You are a nice friend and imagine you have the ability to remember the past conversation and the ability to see the face of the conversation's human. Use the past conversation if it's needed.
                Based on the conversation chat emotion and the human face emotion, alter the responses but note you have to answer their question first if they ask 
                before concerning about his emotion and rank the human chat response within these emotions: sad, fearful, angry, love, embarrassed, happy, neutral.

Use the following format:

Response: respond in a lovely way right here.
Rank the emotion of the human chat input: the ranked value here
.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:'''
)

# Creating Dataframe
df = pd.DataFrame(columns=['encode', 'chat_history'])

# Load emotion detection model from JSON file
with open('emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("emotion_model.h5")

# Function to separate audio from video and extract text
def separate_audio(video_file, audio_path):
    try:
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(audio_path)

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text
    except Exception as e:
        print(f"Error separating audio: {e}")
        return ""

# Function to process camera data
def process_camera(video_file):
    try:
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        audio_path = "output_audio.wav"

        text = separate_audio(video_file, audio_path)

        matches = []
        labels = []
        audios = []

        video = cv2.VideoCapture(video_file)
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 50 == 0:  # Process every 50th frame to speed up
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                for face_encoding in df['encode']:
                    matches.extend(face_recognition.compare_faces(face_encoding, face_encodings))
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    roi_gray = gray_frame[y:y + h, x:x + w]
                    resized_img = cv2.resize(roi_gray, (48, 48))
                    cropped_img = np.expand_dims(np.expand_dims(resized_img, -1), 0)
                    emotion_prediction = emotion_model.predict(cropped_img)
                    max_index = int(np.argmax(emotion_prediction))
                    labels.append(emotion_dict[max_index])
                    print(labels)

        video.release()
        cv2.destroyAllWindows()

        if labels:
            most_common_emotion = Counter(labels).most_common(1)[0][0]
        else:
            most_common_emotion = "Neutral"

        matched_index = -1
        for idx, match in enumerate(matches):
            if True == match:
                print('match found')
                matched_index = idx
                break

        if matched_index == -1:
            print('no match found')
            if face_encodings:
                new_row = {"encode": face_encodings[0], "chat_history": ""}
                df.loc[len(df)] = new_row
                matched_index = len(df) - 1

        combined_input = f"human face emotion: {most_common_emotion} question: {text}"
        chat_history = df.iloc[matched_index]['chat_history']
        
        conversation_with_summary = ConversationChain(
            llm=llm,
            prompt=prompt_template,
            memory=ConversationSummaryBufferMemory(
                llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.6),
                max_token_limit=80000,
                moving_summary_buffer=chat_history
            )
        )
        output = conversation_with_summary.predict(input=combined_input)

        response_match = re.search(r'Response: (.+?)\n', output)
        emotion_match = re.search(r'Rank the emotion of the human chat input: (\w+)', output)
        response = response_match.group(1).strip() if response_match else None
        emotion_word = emotion_match.group(1) if emotion_match else None

        tts_options = TTSOptions(
            voice="s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json",
            sample_rate=24000,
            format=Format.FORMAT_MP3,
            speed=0.7 if emotion_word and emotion_word.lower() in ['sad', 'fearful', 'angry'] else 0.9,
        )
        audio_filename = f"generated_audio{'5' if emotion_word and emotion_word.lower() in ['sad', 'fearful', 'angry'] else '4'}.mp3"
        with open(audio_filename, "wb") as audio_file:
            for chunk in client.tts(text=[response], voice_engine="PlayHT2.0-turbo", options=tts_options):
                audio_file.write(chunk)
        audios.append(audio_filename)

        df.iloc[matched_index, 1] = conversation_with_summary.memory.moving_summary_buffer
        return audios
    except Exception as e:
        print(f"Error processing camera data: {e}")
        return []

print(df)
