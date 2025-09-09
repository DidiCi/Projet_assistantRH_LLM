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

def parse_name(filename):
    """Extract first and last name from CV filename: CV_LASTNAME Firstname.pdf"""
    match = re.match(r"CV_([A-Za-zÀ-ÖØ-öø-ÿ\-]+)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-]+)\.pdf", filename)
    if match:
        last, first = match.groups()
        return first, last
    return None, None


def extract_images_and_fix_markdown(doc, filename, markdown_text):
    """Save images and replace <!-- image --> placeholders with proper links"""
    images_folder = Path(folder_path) / "cv_images"
    images_folder.mkdir(exist_ok=True)

    image_map = {}
    for i, pic in enumerate(doc.document.pictures):
        if pic.image:
            out_path = images_folder / f"{filename}_img_{i}.png"
            with open(out_path, "wb") as f:
                f.write(pic.image.png_bytes())
            image_map[i] = out_path.name
            print(f"✅ Image sauvegardée: {out_path}")

    fixed_text = markdown_text
    for i, path in image_map.items():
        fixed_text = fixed_text.replace("<!-- image -->", f"![image]({path})", 1)

    return fixed_text


def split_by_sections(full_text):
    """Split CV text into semantic sections"""
    # Look for typical CV section headings
    section_pattern = re.compile(
        r"(?P<header>^(?:PROFIL|PRÉSENTATION|COMP[ÉE]TENCES|EXP[ÉE]RIENCES?|FORMATION|LANGUES?|LOISIRS|CENTRES D[’']?INTERETS?))",
        re.IGNORECASE | re.MULTILINE
    )

    chunks = []
    matches = list(section_pattern.finditer(full_text))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        section_name = match.group("header").strip()
        section_text = full_text[start:end].strip()
        chunks.append((section_name.upper(), section_text))

    # fallback: if nothing matched, return entire text as one section
    if not chunks:
        chunks = [("FULL", full_text)]

    return chunks


def split_with_fallback(section_text, section_name, filename, first, last):
    """Recursive splitter for large sections"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_text(section_text)

    return [
        {
            "text": split,
            "source": filename,
            "first_name": first,
            "last_name": last,
            "section": section_name
        }
        for split in splits
    ]


def get_chunks():
    converter = DocumentConverter()
    all_chunks = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Traitement de {filename}...")

            # Parse name from filename
            first, last = parse_name(filename)

            # Extract text with Docling
            doc = converter.convert(file_path)
            full_text = doc.document.export_to_markdown()

            # Fix images in markdown
            full_text = extract_images_and_fix_markdown(doc, filename, full_text)

            # Split into semantic sections
            sections = split_by_sections(full_text)

            # Chunk each section
            for section_name, section_text in sections:
                section_chunks = split_with_fallback(section_text, section_name, filename, first, last)
                all_chunks.extend(section_chunks)

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
