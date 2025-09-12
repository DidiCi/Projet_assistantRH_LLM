import os
import json
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Directory with PDFs
pdf_folder = "/home/diletta.ciardo@Digital-Grenoble.local/FormationIA/15. LLM/data/raw"
pdf_folder_out = "/home/diletta.ciardo@Digital-Grenoble.local/FormationIA/15. LLM/data/structured"
all_chunks = []

# Loop over all PDFs
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        file_path = os.path.join(pdf_folder, pdf_file)
        
        # Extract text from PDF
        elements = partition(file_path, languages=['fr'])
        text = "\n".join([str(e) for e in elements])
        # Chunk the text
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        
        # Create json objects for each chunk
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": pdf_file,
                "chunk_id": i
            })

# Save all chunks to a JSON file
file_output = os.path.join(pdf_folder_out, "all_chunks.json")
print(file_output)
with open(file_output, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Processed {len(all_chunks)} chunks from {len(os.listdir(pdf_folder))} PDFs.")
