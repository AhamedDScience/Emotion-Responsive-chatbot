# Project Title

## Overview

**This project integrates advanced AI components to create a conversational agent capable of recognizing user emotions, personalizing responses, and generating human-like audio replies. The core functionalities revolve around a large language model (LLM), emotion detection, face recognition, and sophisticated data handling to ensure contextually aware interactions.**

## Components

## Large Language Model (LLM)
+ Model: Google Gemini-1.5-pro-latest

+ Function: Provides the core conversational capabilities of the agent, ensuring fluid and coherent dialogues.
  
## Framework

+ Langchain: Facilitates the seamless integration of different components, enabling a cohesive interaction flow between the LLM, emotion detection, and face recognition modules.

## Emotion Detection

+ Model: Convolutional Neural Network (CNN)

+ Function: Identifies user emotions through facial expressions, allowing the system to understand and respond appropriately to the user's emotional state.

## Face Recognition

+ Library: Utilized to capture and encode user faces for personalization.

+ Function: Enables the system to store and retrieve past user data based on face encodings, enhancing the personalization of responses.

## Data Storage

+ Method: Past user data is stored based on face encodings.

+ Function: Allows the system to provide contextual responses by referring to previous interactions.

## Prompt Engineering

+ Framework: Langchain prompt templates

+ Function: Guides the LLM in processing user inputs, emotion predictions, and past interaction data to generate appropriate responses.

# Emotional Response Generation

+ Process: The LLM evaluates the ranked emotions, considers the facial emotion detection, and incorporates past data to formulate a response with a concluding
  emotion.

+ Goal: To provide emotionally intelligent responses that resonate with the user's current emotional state.

## Text-to-Speech (TTS)

+ Function: Generates audio responses that align with the concluding emotion, ensuring a natural and empathetic conversational experience.

## Unique Data Handling Approach
 
+ ConversationSummaryBufferMemory: Used to integrate the LLM into the chat history.
+ Purpose: Not used as a database but to maintain and manage the token limit, storing data coming from the ConversationSummaryBufferMemory.
