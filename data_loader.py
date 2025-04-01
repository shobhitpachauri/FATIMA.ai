import json
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login

# Directly login with token
login(token="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")

# Debug print
print("Environment variables loaded")
print("HUGGINGFACE_TOKEN value:", os.getenv("HUGGINGFACE_TOKEN"))

# Load JSON data
DATA_FILE = "scrap_data.json"
INDEX_FOLDER = "faiss_index"

def load_json_data():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Your data is a dictionary with URLs as keys and text as values
    # Let's extract just the text content
    return list(data.values())

# Process & store in FAISS
def create_vector_db():
    texts = load_json_data()

    # Split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]  # Added separators for better splitting
    )
    docs = text_splitter.create_documents(texts)

    # Convert to embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vector_db = FAISS.from_documents(docs, embedding_model)

    # Save database
    vector_db.save_local(INDEX_FOLDER)
    print(f"Vector database created & stored in '{INDEX_FOLDER}/'")

if __name__ == "__main__":
    create_vector_db()
