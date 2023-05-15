import os
import openai
import sounddevice as sd
import tempfile
import numpy as np
from scipy.io import wavfile
from scipy.spatial.distance import cosine
import json
# from flask import Flask, request, jsonify
#
# app = Flask(__name__)

# Prompt the user for the OpenAI API key
api_key = input("Enter your OpenAI API key: ")
openai.api_key = api_key
embeddings_folder="./embeddings"
# Prompt the user for their name or nickname
name = input('Enter Your Name or Nickname: ')

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    embedding = np.squeeze(response['data'][0]['embedding'])
    return embedding

def search(query, embedding_dict):
    query_embedding = get_embedding(query, model="text-embedding-ada-002")
    query_embedding = np.squeeze(query_embedding)
    results = []

    for name, item in embedding_dict.items():
        transcript_embedding = np.squeeze(item["embedding"])
        similarity = 1 - cosine(query_embedding, transcript_embedding)
        results.append((name, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


# Set the desired audio parameters
sample_rate = 16000  # Sample rate in Hz
duration = 10  # Recording duration in seconds

# Store the embeddings in a local file (JSON format)
# Load the embeddings from the file (if it exists and is not empty)
embedding_file = "embeddings.json"
embedding_dict = {}  # Define the embedding dictionary

# Check if the embeddings file exists
if os.path.isfile(embedding_file):
    # Load the embeddings from the file
    with open(embedding_file, "r") as file:
        try:
            embedding_dict_json = json.load(file)
            if embedding_dict_json:  # Check if the file is not empty
                # Convert the embeddings from lists to NumPy arrays
                embedding_dict = {key: np.array(embedding) for key, embedding in embedding_dict_json.items()}
            else:
                print("Embeddings file is empty.")
        except json.JSONDecodeError:
            # Invalid JSON format, handle the error
            print("Invalid JSON format in embeddings file.")
else:
    # File does not exist, create a new embeddings dictionary
    print("Embeddings file not found. Creating a new embeddings dictionary.")

# Save the initial embeddings dictionary to the JSON file
with open(embedding_file, "w") as file:
    # Convert the embeddings from lists to NumPy arrays before saving
    embedding_dict_json = {key: embedding.tolist() for key, embedding in embedding_dict.items()}
    json.dump(embedding_dict_json, file)

# Start the chat loop

# Start the chat loop
while True:
    # Record audio from the microphone or accept text query
    query_option = input("Enter '1' to provide a voice query, '2' to provide a text query: ")
    if query_option == '1':
        # Record audio from the microphone
        print("Recording audio. Speak into the microphone...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()

        # Save the recorded audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.name  # Path to the temporary audio file
            wavfile.write(temp_audio.name, sample_rate, audio)

        # Transcribe the audio using OpenAI API
        with open(temp_audio.name, "rb") as audio_file:
            response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )

            # Get the transcript from the response
            transcript = response['text']
            print("Transcript:")
            print(transcript)
    elif query_option == '2':
        transcript = input("Enter your text query: ")
    else:
        print("Invalid option. Please try again.")
        continue

    # Embed the query and store it locally
    embedding = get_embedding(transcript)
    embedding_dict[name] = {
        "transcript": transcript,
        "embedding": embedding.tolist()  # Convert NumPy array to list
    }

    # Save the updated embeddings to the JSON file
    with open("./embeddings.json", "w") as file:
        # Convert the embeddings from lists to NumPy arrays before saving
        embedding_dict_json = {key: np.array(embedding).tolist() for key, embedding in embedding_dict.items()}
        json.dump(embedding_dict_json, file)

    # Perform search with the query and relevant documents
    search_results = search(transcript, embedding_dict)
    print("Search Results:")
    for name, similarity in search_results:
        print(f"{name}: {similarity}")

    # Concatenate relevant documents and conversation history into messages
    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
        {"role": "user", "content": f"What documents should I consider for this query? {transcript}"}
    ]

    # Load any additional relevant documents from the "embeddings" folder
    if os.path.isdir(embeddings_folder):
        for filename in os.listdir(embeddings_folder):
            if filename.endswith(".json"):
                with open(os.path.join(embeddings_folder, filename), "r") as file:
                    additional_embeddings = json.load(file)
                    embedding_dict.update(additional_embeddings)

    # Submit the query with the concatenated messages
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        max_tokens=50
    )

    # Retrieve the assistant's response
    assistant_response = response['choices'][0]['message']['content']

    # Print and store the assistant's response
    print("Assistant's Response:")
    print(assistant_response)

    # Perform search with the assistant's response and relevant documents
    search_results = search(assistant_response, embedding_dict)
    print("Search Results:")
    for name, similarity in search_results:
        print(f"{name}: {similarity}")

    # Ask GPT to summarize the conversation
    summary_prompt = "Summarize this conversation between helpful AI bot and user:"
    messages.append({"role": "assistant", "content": assistant_response})
    messages_summary = messages.copy()
    messages_summary.append({"role": "system", "content": summary_prompt})

    response_summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_summary,
        temperature=0.8,
        max_tokens=50
    )

    conversation_summary = response_summary['choices'][0]['message']['content']
    print("Conversation Summary:")
    print(conversation_summary)

    # Ask GPT for a response using the conversation summary and user's question
    response_prompt = f"{conversation_summary}\n\nUser: {transcript}"
    messages.append({"role": "system", "content": response_prompt})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.8,
        max_tokens=50
    )

    # Retrieve the assistant's response
    assistant_response = response['choices'][0]['message']['content']

    # Print and store the assistant's response
    print("Assistant's Response:")
    print(assistant_response)

    # Perform search with the assistant's response and relevant documents
    search_results = search(assistant_response, embedding_dict)
    print("Search Results:")
    for name, similarity in search_results:
        print(f"{name}: {similarity}")

    # Allow the user to ask more questions
    more_questions = input("Do you want to ask more questions? (y/n): ").lower() == 'y'

    if more_questions:
        continue
    else:
        break
