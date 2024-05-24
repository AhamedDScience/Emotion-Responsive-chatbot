import streamlit as st
from main import process_camera
def main():
    st.title("Emotion-Responsive ChatBot")
    
    # Upload video file
    video_file = st.file_uploader("Upload Video", type=["mp4", "mov"])
    
    if video_file is not None:
        
        # Show loading message while processing
        with st.spinner():
            # Call function to process video and get audio
            val = process_camera(video_file.name)
            audio_file = val[0]
        
        # Display audio file and provide option to play it
        st.audio(audio_file, format='audio/wav', start_time=0)
        

main()
