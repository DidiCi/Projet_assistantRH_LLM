import os
import json
import sys
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
LANGSMITH_TRACING = os.environ.get("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ocr_agent = os.environ.get("OCR_AGENT", "tesseract") 


# Folder containing PDFs
folder_path = "/home/jip.wulffele@Digital-Grenoble.local/Documents/15_LLM/project_llm/CVthèque/"

# Initialize chunker
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

# Loop over all PDFs in the folder
all_chunks = []
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing {filename}...")

        # Extract text
        document = partition_pdf(filename=file_path,
                                 #strategy="hi_res",
                                 ocr=True,
                                 #extract_images_in_pdf=True,                           
                                 #extract_image_block_types=["Image", "Table"],         
                                 #extract_image_block_to_payload=False,
                                 ocr_language=["eng", "fra"])
        full_text = " ".join([d.text for d in document])

        # Chunk text
        chunks = text_splitter.split_text(full_text)

        # Store chunks with metadata (filename)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": filename
            })

print(f"Processed {len(all_chunks)} chunks from {len(os.listdir(folder_path))} PDFs.")

output_file = os.path.join(folder_path, "cv_chunks.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Saved {len(all_chunks)} chunks to {output_file}")

# Embeddings avec Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Chroma local
persist_directory = os.path.join(folder_path, "chroma_db")

vectorstore = Chroma(
    collection_name="cv_collection",
    embedding_function=embeddings,
    persist_directory=persist_directory
)

ids = vectorstore.add_documents(documents=all_chunks)

texts = [chunk["text"] for chunk in all_chunks]
metadatas = [{"source": chunk["source"]} for chunk in all_chunks]

vectorstore.add_texts(texts=texts, metadatas=metadatas)

print(f"Base vectorielle Chroma (Gemini embeddings) créée dans {persist_directory}")

sys.exit(0)
