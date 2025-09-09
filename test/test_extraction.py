import os, json, re
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from docling.document_converter import DocumentConverter
from pathlib import Path

######################## Charger les variables d'environnement
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

######################## Dossiers
folder_path = "/home/jip.wulffele@Digital-Grenoble.local/Documents/15_LLM/project_llm/CVthèque_small/"

chuncks_out = "cv_chunks.json"

folder_chroma = "chroma_db"
collection_chroma = "cv_collection"

######################## Hyperparameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SPLIT_BY_HEADER = True

MODEL_EMBEDDINGS = "models/embedding-001"

######################## Functions

def extract_images(doc, filename):
        images_folder = Path(folder_path) / "cv_images"
        images_folder.mkdir(exist_ok=True)

        for i, pic in enumerate(doc.document.pictures):  # <- use .pictures
            if pic.image:
                out_path = images_folder / f"{filename}_img_{i}.png"
                with open(out_path, "wb") as f:
                    f.write(pic.image.png_bytes())
                print(f"✅ Image sauvegardée: {out_path}")


def parse_name(filename):
    match = re.match(r"CV_([A-Za-zÀ-ÖØ-öø-ÿ\-]+)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-]+)\.pdf", filename)
    if match:
        last, first = match.groups()
        return first, last
    return None, None



def get_chunks(split_by_header=False):

    if split_by_header == True:
        # Initialiser le chunker
        text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "title"),
                #("##", "section"),
                #("###", "subsection")
            ],
            return_each_line=True,
            strip_headers=False
        )
    else:
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

            # Extraire texte et images avec Docling
            doc = converter.convert(file_path)
            extract_images(doc, filename) 
            full_text = doc.document.export_to_markdown()  # Markdown = + lisible que du brut
            print(full_text)
            # Découper en chunks
            chunks = text_splitter.split_text(full_text)

            # Get nom + prenom
            first, last = parse_name(filename)

            # Sauvegarder avec métadonnées
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk if isinstance(chunk, str) else chunk.page_content,
                    "source": filename,
                    "first_name": first,
                    "last_name": last
                })

    print(f"{len(all_chunks)} chunks extraits depuis {len(os.listdir(folder_path))} PDFs.")

    return all_chunks

def main():
    all_chunks = get_chunks(split_by_header=SPLIT_BY_HEADER)

######################## Main
if __name__ == "__main__":
    main()
