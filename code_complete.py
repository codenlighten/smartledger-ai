import os
import openai
import json
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import faiss

api_key = input("Enter your OpenAI API key: ")
openai.api_key = api_key

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def complete_code(text, top_result):
    try:
        prompt = f"provide the complete working code for the following:\n{text}\n\nConsider the following related item: {top_result}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=4096,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["###"]
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_embeddings_and_index(embedding_file):
    with open(embedding_file, "r") as file:
        data = json.load(file)
    embeddings = [item['embedding'] for item in data]
    chunks = [item['chunk'] for item in data]
    embeddings_np = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return embeddings_np, index, chunks

def vector_search(query, index, embeddings, chunks):
    # Generate an embedding for the query
    query_embedding = model.encode(query)

    # Perform a nearest neighbor search
    D, I = index.search(np.array([query_embedding]), 1)

    # I[0][0] gives the index of the nearest neighbor
    nearest_neighbor = chunks[I[0][0]]
    print(nearest_neighbor)
    return nearest_neighbor


query = input('What is your code question?')

# Load embeddings and perform vector search
embeddings, index = load_embeddings_and_index("./embeddings/new/embeddings.json")
top_result = vector_search(query, index, embeddings)
# print(top_result)
# Call the complete_code function with the top search result
response = complete_code(query, top_result)
print(response)

# Ask the user for feedback
feedback = input("Are you satisfied with the response? (yes/no): ")

# If the user is satisfied, store the conversation
if feedback.lower() == "yes":
    conversation = {
        "query": query,
        "top_result": top_result,
        "response": response
    }

    # Ensure the output directory exists
    os.makedirs("./embeddings/conversations", exist_ok=True)

    # Load existing conversations or create a new list
    if os.path.exists("./embeddings/conversations/embeddings.json"):
        with open("./embeddings/conversations/embeddings.json", "r") as file:
            conversations = json.load(file)
    else:
        conversations = []

    # Add the new conversation to the list
    conversations.append(conversation)

    # Save the conversations to a file
    with open("./embeddings/conversations/embeddings.json", "w") as file:
        json.dump(conversations, file)
