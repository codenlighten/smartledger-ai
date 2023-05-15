import os
import glob
import json
import numpy as np
import faiss
import ast
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

def process_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Failed to read file {filename}: {e}")
        return []

    try:
        if filename.endswith('.py'):
            chunks = split_python_file(content)
        elif filename.endswith('.ts') or filename.endswith('.js'):
            chunks = split_typescript_or_javascript_file(content)
        elif filename.endswith('.md'):
            chunks = re.split(r'\n#+ ', content)
        elif filename.endswith('.txt'):
            chunks = re.split(r'\n\s*\n', content)
        else:
            chunks = content.split('\n')
    except Exception as e:
        print(f"Failed to split file {filename}: {e}")
        return []

    return chunks

def split_typescript_or_javascript_file(content):
    lines = content.split('\n')
    chunks = ['\n'.join(lines[i:i+10]) for i in range(0, len(lines), 5)]
    return chunks

def split_python_file(content):
    # Parse the Python code into an AST
    module = ast.parse(content)

    # Initialize a list to store the chunks
    chunks = []

    # Iterate over the top-level statements in the module
    for statement in module.body:
        # If the statement is a function or class definition, add it to the chunks
        if isinstance(statement, (ast.FunctionDef, ast.ClassDef)):
            # Get the source lines of the statement
            lines = content.splitlines()[statement.lineno - 1:statement.end_lineno]

            # Join the lines back together and add to the chunks
            chunks.append('\n'.join(lines))

    return chunks

# Load processed files from a file
if os.path.exists('processed_files.txt'):
    with open('processed_files.txt', 'r') as f:
        processed_files = set(line.strip() for line in f)

# Parse over the directory tree
for file_type in file_types:
    for filename in glob.glob(os.path.join(directory, '**', file_type), recursive=True):
        if filename in processed_files:
            continue

        chunks.extend(process_file(filename))

        processed_files.add(filename)

# Save the processed files to a file
with open('processed_files.txt', 'w') as f:
    for filename in processed_files:
        f.write(filename + '\n')

# Generate embeddings for each chunk
try:
    embeddings = model.encode(chunks)
except Exception as e:
    print(f"Failed to generate embeddings: {e}")
    embeddings = []

# Combine chunks and embeddings into a list of dictionaries
data = [{'chunk': chunk, 'embedding': embedding.tolist()} for chunk, embedding in zip(chunks, embeddings)]

# Convert embeddings to a numpy array
embeddings_np = np.array([item['embedding'] for item in data])
# Build a FAISS index for nearest neighbor search
index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

# Define the output directory
output_directory = './embeddings/new'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Save to JSON file
with open(os.path.join(output_directory, 'embeddings.json'), 'w') as f:
    json.dump(data, f)

# Save the FAISS index
faiss.write_index(index, os.path.join(output_directory, 'index.faiss'))

# Define the query
while True:
    query = input("Enter your code question (or 'quit' to exit): ")

    # Check if the user wants to quit
    if query.lower() == 'quit':
        break

    # Generate an embedding for the query
    query_embedding = model.encode([query])[0]

    # Perform a nearest neighbor search
    D, I = index.search(np.array([query_embedding]), 1)

    # I[0][0] gives the index of the nearest neighbor
    nearest_neighbor = data[I[0][0]]['chunk']

    print("Nearest neighbor to the query:", nearest_neighbor)
