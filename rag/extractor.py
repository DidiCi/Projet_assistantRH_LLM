import os
import json
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import PDF_INPUT_FOLDER, PDF_OUTPUT_FOLDER

class PDFExtractor:
    def __init__(self, input_folder=PDF_INPUT_FOLDER, output_folder=PDF_OUTPUT_FOLDER, chunk_size=500, chunk_overlap=50):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Ensure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)

    def extract_chunks(self):
        all_chunks = []
        pdf_files = [f for f in os.listdir(self.input_folder) if f.endswith(".pdf")]
        
        for pdf_file in pdf_files:
            file_path = os.path.join(self.input_folder, pdf_file)

            # Extract text from PDF
            elements = partition(file_path, languages=['fr'])
            text = "\n".join([str(e) for e in elements])

            # Chunk the text
            splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            chunks = splitter.split_text(text)

            # Create json objects for each chunk
            for i, chunk in enumerate(chunks):
                all_chunks.append({"text": chunk, "source": pdf_file, "chunk_id": i})

        # Save all chunks to the path output_file="output_folder/all_chunks.json"
        output_file = os.path.join(self.output_folder, "all_chunks.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"Processed {len(all_chunks)} chunks from {len(pdf_files)} PDFs.")
        return output_file
