import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import os
from langchain.schema import Document
import time 
import shutil

pdf_folder = "/home/diletta.ciardo@Digital-Grenoble.local/FormationIA/15. LLM/data"

# Set API key for Google Gemini
from config import GOOGLE_API_KEY

# Load chuncks from JSON file
with open(pdf_folder + "/structured/all_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Convert to LangChain Document objects
all_splits = [
    Document(page_content=chunk["text"], metadata={
        "source": chunk.get("source", "unknown"),
        "chunk_id": chunk.get("chunk_id", "unknown"),
    })
    for chunk in chunks
]

# Model for embedding
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Create Chroma vectorstore
shutil.rmtree("./chroma_langchain_db")  # Clear previous DB if exists
vector_store = Chroma(
    collection_name="hr_chunks",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

# Batch ingestion to avoid quota issues
BATCH_SIZE = 5
for i in range(0, len(all_splits), BATCH_SIZE):
    batch = all_splits[i:i + BATCH_SIZE]
    vector_store.add_documents(documents=batch)
    print(f"Added batch {i // BATCH_SIZE + 1} ({len(batch)} documents)")
    time.sleep(5)  # wait 1 second to avoid hitting Gemini rate limits

print(f"{len(all_splits)} chunks ingested into Chroma (LangChain wrapper)")
