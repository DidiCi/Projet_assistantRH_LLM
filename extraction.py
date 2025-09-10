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

from langchain_mistralai import MistralAIEmbeddings

from pathlib import Path

######################## Charger les variables d'environnement
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

######################## Dossiers
folder_path = "/home/jip.wulffele@Digital-Grenoble.local/Documents/15_LLM/project_llm/CVthèque/"

chuncks_out = "cv_chunks_by_header_mistral.json"

folder_chroma = "chroma_db_by_header_mistral"
collection_chroma = "cv_collection"

######################## Hyperparameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

IMAGE_RESOLUTION_SCALE = 2.0

SPLIT_BY_HEADER = True
ONE_CHUNK_CV = False

#MODEL_EMBEDDINGS = "models/embedding-001"
MODEL_EMBEDDINGS = "mistral-embed"

######################## Functions

def get_chunks(split_by_header=False, one_chunk_cv=False):


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

            # Get nom + prenom
            first, last = parse_name(filename)

            if one_chunk_cv == True:
                # ✅ Sauvegarder 1 CV complet comme un seul chunk
                all_chunks.append({
                    "text": full_text.strip(),
                    "source": filename,
                    "first_name": first,
                    "last_name": last
                })
            else:

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

                # Découper en chunks
                chunks = text_splitter.split_text(full_text)

                # Sauvegarder avec métadonnées
                for chunk in chunks:
                    text = chunk if isinstance(chunk, str) else chunk.page_content
                    # Skip image placeholders or empty chunks
                    if text.strip() and text.strip() != "<!-- image -->":
                        all_chunks.append({
                            "text": chunk if isinstance(chunk, str) else chunk.page_content,
                            "source": filename,
                            "first_name": first,
                            "last_name": last
                        })

    print(f"{len(all_chunks)} chunks extraits depuis {len(os.listdir(folder_path))} files.")

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
    #embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDINGS)
    embeddings = MistralAIEmbeddings(model=MODEL_EMBEDDINGS)

    # Chroma local
    persist_directory = os.path.join(folder_path, folder_chroma)

    vectorstore = Chroma(
        collection_name=collection_chroma,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    texts = [chunk["text"] for chunk in all_chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "first_name": chunk["first_name"],
            "last_name": chunk["last_name"]
        }
        for chunk in all_chunks]

    vectorstore.add_texts(texts=texts, metadatas=metadatas) # Emdeeing applied at this step

    print(f"Base vectorielle Chroma (Gemini embeddings) créée dans {persist_directory}")


def main():
    all_chunks = get_chunks(split_by_header=SPLIT_BY_HEADER, one_chunk_cv=ONE_CHUNK_CV)
    save_chunks(all_chunks)

    create_vectorstore(all_chunks)


######################## Main
if __name__ == "__main__":
    main()
