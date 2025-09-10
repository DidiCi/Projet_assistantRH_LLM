from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# Set API key for Google Gemini
from config import GOOGLE_API_KEY

# Initialize the same embeddings as before
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load the existing Chroma vector store
vector_store = Chroma(
    collection_name="hr_chunks",      # same collection name
    embedding_function=embeddings,    # same embeddings
    persist_directory="./chroma_langchain_db"  # same folder
)

# Create a retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Example query
query = "Quelles langues parle Ronan ?"
results = retriever.get_relevant_documents(query)

for doc in results:
    print("-----")
    print(doc)


