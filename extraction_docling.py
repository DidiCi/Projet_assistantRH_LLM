import os, json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from docling.document_converter import DocumentConverter

# Charger les variables d'environnement
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Dossier contenant les PDFs
folder_path = "/home/jip.wulffele@Digital-Grenoble.local/Documents/15_LLM/project_llm/CVthèque/"

# Initialiser le chunker
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# Initialiser Docling
converter = DocumentConverter()

# Parcourir tous les PDFs
all_chunks = []
for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        print(f"Traitement de {filename}...")

        # Extraire texte avec Docling
        doc = converter.convert(file_path)
        full_text = doc.document.export_to_markdown()  # Markdown = + lisible que du brut

        # Découper en chunks
        chunks = text_splitter.split_text(full_text)

        # Sauvegarder avec métadonnées
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": filename
            })

print(f"{len(all_chunks)} chunks extraits depuis {len(os.listdir(folder_path))} PDFs.")

# Sauvegarder aussi en JSON (optionnel)
output_file = os.path.join(folder_path, "cv_chunks.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Chunks sauvegardés dans {output_file}")

# Embeddings avec Gemini
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Chroma local
persist_directory = os.path.join(folder_path, "chroma_db")

vectorstore = Chroma(
    collection_name="cv_collection",
    embedding_function=embeddings,
    persist_directory=persist_directory
)

texts = [chunk["text"] for chunk in all_chunks]
metadatas = [{"source": chunk["source"]} for chunk in all_chunks]

vectorstore.add_texts(texts=texts, metadatas=metadatas)

print(f"Base vectorielle Chroma (Gemini embeddings) créée dans {persist_directory}")
