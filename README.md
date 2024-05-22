# Emotion-Responsive-LLM
“This project aimed to develop an emotion-responsive chatbot capable of personalizing interactions based on a user's emotional state and past conversations.”

![chatbot](https://github.com/AhamedDScience/Emotion-Responsive-LLM/assets/167436292/cec3a5bd-52b4-4745-93f9-b51c544a13b5)


Large Language Model (LLM):
Google gemini-1.5-pro-latest provided the core conversational capabilities.

Framework:
Langchain facilitated the integration of different components.

Emotion Detection:
A Convolutional Neural Network (CNN) model identified emotions by user face.

Face Recognition:
A face recognition library captured user encodings for personalization.  

Data Storage:
Past user data was stored based on face encodings, enabling contextual responses.

Prompt Engineering:
Langchain prompt templates guided the LLM to process user input, emotion predictions, and past data.

Emotional Response Generation:
The LLM ranked emotions, considered face-based emotion, and past data to formulate an appropriate response with a concluding emotion.

Text-to-Speech (TTS):
A function generated audio responses tailored to the concluding emotion.
