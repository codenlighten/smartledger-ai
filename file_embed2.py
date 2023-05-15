import os
import glob
import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the directory to parse
directory = './docs2vectorize'

# Define the file types to process
file_types = ['*.py', '*.ts', '*.js', '*.md', "*.txt"]

# Initialize a list to store all chunks
chunks = []
processed_files = set()

# Load processed files from a file
if os.path.exists('processed_files.txt'):
    with open('processed_files.txt', 'r') as f:
        processed_files = set(line.strip() for line in f)

# Parse over the directory tree
for file_type in file_types:
    for filename in glob.glob(os.path.join(directory, '**', file_type), recursive=True):
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()

            # Split the content into chunks based on the file type
            if filename.endswith('.py'):
                # Split Python files by function and class definitions
                # (you would need to implement this)
                file_chunks = split_python_file(content)
            elif filename.endswith('.ts') or filename.endswith('.js'):
                # Split TypeScript and JavaScript files by function and class definitions
                # (you would need to implement this)
                file_chunks = split_typescript_or_javascript_file(content)
            elif filename.endswith('.md'):
                # Split Markdown files by section headings
                file_chunks = re.split(r'\n#+ ', content)
            elif filename.endswith('.txt'):
                # Split text files by blank lines
                file_chunks = re.split(r'\n\s*\n', content)
            else:
                # If the file type is not recognized, just split by lines
                file_chunks = content.split('\n')

            chunks.extend(file_chunks)
    for filename in glob.glob(os.path.join(directory, '**', file_type), recursive=True):
        # Skip the file if it has already been processed
        if filename in processed_files:
            continue


        # Add the file to the set of processed files
        processed_files.add(filename)

# Save the processed files to a file
with open('processed_files.txt', 'w') as f:
    for filename in processed_files:
        f.write(filename + '\n')

# Generate embeddings for each chunk
embeddings = [model.encode(chunk)[np.newaxis, :] for chunk in chunks]

# Convert embeddings to a numpy array
embeddings_np = np.concatenate(embeddings, axis=0)

# Combine chunks and embeddings into a list of dictionaries
data = [{'chunk': chunk, 'embedding': embedding.tolist()} for chunk, embedding in zip(chunks, embeddings)]

# Define the output directory
output_directory = './embeddings/new'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Save to JSON file
with open(os.path.join(output_directory, 'embeddings.json'), 'w') as f:
    json.dump(data, f)

# Build a FAISS index for nearest neighbor search
embeddings_np = np.array([item['embedding'] for item in data])
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

# Define the query

while True:
    query = input("Enter your code question (or 'quit' to exit): ")

    # Check if the user wants to quit
    if query.lower() == 'quit':
        break

    # Generate an embedding for the query
    query_embedding = model.encode(query)

    # Perform a nearest neighbor search
    D, I = index.search(np.array([query_embedding]), 1)

    # I[0][0] gives the index of the nearest neighbor
    nearest_neighbor = data[I[0][0]]['chunk']

    print("Nearest neighbor to the query:", nearest_neighbor)


