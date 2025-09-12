import json
import shutil
import time
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY, CHROMA_DB_PATH

class VectorStoreIngestor:
    def __init__(self, chunk_json_path, collection_name="hr_chunks", batch_size=5):
        self.chunk_json_path = chunk_json_path
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    def ingest(self):

        # Load chuncks from JSON file
        with open(self.chunk_json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        # Convert to LangChain Document objects
        documents = [
            Document(page_content=chunk["text"], metadata={"source": chunk.get("source", "unknown"), "chunk_id": chunk.get("chunk_id", "unknown")})
            for chunk in chunks
        ]

        # Clear previous DB if exists
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        # Create Chroma vectorstore
        vector_store = Chroma(collection_name=self.collection_name, embedding_function=self.embeddings, persist_directory=CHROMA_DB_PATH)

        # Batch ingestion to avoid quota issues
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            vector_store.add_documents(documents=batch)
            print(f"Added batch {i // self.batch_size + 1} ({len(batch)} documents)")
            time.sleep(5)

        print(f"{len(documents)} chunks ingested into Chroma.")
        return vector_store
