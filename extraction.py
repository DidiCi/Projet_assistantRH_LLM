import os, json, re

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem


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

IMAGE_RESOLUTION_SCALE = 2.0

SPLIT_BY_HEADER = True

MODEL_EMBEDDINGS = "models/embedding-001"

######################## Functions

def get_chunks(split_by_header=False):

    if split_by_header == True:
        # Initialiser le chunker
        text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "title"),
                ("##", "section"),
                #("###", "subsection")
            ],
            return_each_line=False,
            strip_headers=False
        )
    else:
        # Initialiser le chunker
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    # Initialiser Docling
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    #converter = DocumentConverter()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

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


def extract_images(doc, filename):
    images_folder = Path(os.path.join(folder_path, "cv_images")) / filename
    images_folder.mkdir(parents=True, exist_ok=True)

    if hasattr(doc.document, "pictures"):
        for i, pic in enumerate(doc.document.pictures):
            if pic.image:
                out_path = images_folder / f"{filename}_img_{i}.png"
                pic.image.pil_image.save(out_path, format="PNG")  # ✅ use pil_image
                print(f"✅ Image saved: {out_path}")
    else:
        print(f"⚠️ No pictures found in {filename}")


def parse_name(filename):
    match = re.match(r"CV_([A-Za-zÀ-ÖØ-öø-ÿ\-]+)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-]+)\.pdf", filename)
    if match:
        last, first = match.groups()
        return first, last
    return None, None


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
    all_chunks = get_chunks(split_by_header=SPLIT_BY_HEADER)
    save_chunks(all_chunks)

    create_vectorstore(all_chunks)


######################## Main
if __name__ == "__main__":
    main()
