import os, json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from docling.document_converter import DocumentConverter

######################## Charger les variables d'environnement
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

######################## Dossiers
folder_path = "/home/jip.wulffele@Digital-Grenoble.local/Documents/15_LLM/project_llm/CVthèque/"

chuncks_out = "cv_chunks.json"

folder_chroma = "chroma_db"
collection_chroma = "cv_collection"

######################## Hyperparameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

MODEL_EMBEDDINGS = "models/embedding-001"

######################## Functions

def get_chunks():

    # Initialiser le chunker
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
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

    return all_chunks

def save_chunks(all_chunks):
    # Sauvegarder aussi en JSON (optionnel)
    output_file = os.path.join(folder_path, chuncks_out)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Chunks sauvegardés dans {output_file}")


def create_vectorstore(all_chunks):
    # Embeddings avec Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDINGS)

    # Chroma local
    persist_directory = os.path.join(folder_path, folder_chroma)

    vectorstore = Chroma(
        collection_name=collection_chroma,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    texts = [chunk["text"] for chunk in all_chunks]
    metadatas = [{"source": chunk["source"]} for chunk in all_chunks]

    vectorstore.add_texts(texts=texts, metadatas=metadatas) # Emdeeing applied at this step

    print(f"Base vectorielle Chroma (Gemini embeddings) créée dans {persist_directory}")


def main():
    all_chunks = get_chunks()
    save_chunks(all_chunks)

    create_vectorstore(all_chunks)

######################## Main
if __name__ == "__main__":
    main()
