import os
from dotenv import load_dotenv

load_dotenv()

PDF_INPUT_FOLDER = os.getenv("PDF_INPUT_FOLDER", "../data/raw")
PDF_OUTPUT_FOLDER = os.getenv("PDF_OUTPUT_FOLDER", "../data/structured")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH = "./chroma_langchain_db"