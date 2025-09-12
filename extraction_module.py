import os, re, json
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter


class CVExtractor:
    def __init__(self, config):
        self.config = config
        pipeline_options = PdfPipelineOptions(
            images_scale=config.image_resolution_scale,
            generate_page_images=True,
            generate_picture_images=True
        )
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def extract_chunks(self, file_paths=None):
        """
        Extract chunks from given file paths (list of PDFs).
        If no file_paths is given, process everything in folder_path (old behavior).
        """
        all_chunks = []
        if file_paths is None:
            file_paths = [
                os.path.join(self.config.folder_path, f)
                for f in os.listdir(self.config.folder_path)
                if f.endswith(".pdf")
            ]

        for path in file_paths:
            filename = os.path.basename(path)
            doc = self.converter.convert(path)
            text = doc.document.export_to_markdown()
            chunks = self.split_chunks(text)
            first, last = self._parse_name(filename)

            for chunk in chunks:
                text_content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                all_chunks.append({
                    "text": text_content,
                    "source": filename,
                    "first_name": first,
                    "last_name": last
                })
        return all_chunks
    
    def split_chunks(self, text):
        if self.config.one_chunk_cv == True:
            return [text] # return as list for consistency
        elif self.config.split_by_header == True:
            parts = re.split(r'(?=\n\n## )', text)
            return [p.strip() for p in parts if p.strip()]
        elif self.config.one_chunk_cv == False:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            return text_splitter.split_text(text) # Découper en chunks


    def _parse_name(self, filename):
        match = re.match(r"CV_([A-Za-zÀ-ÖØ-öø-ÿ\-]+)\s+([A-Za-zÀ-ÖØ-öø-ÿ\-]+)\.pdf", filename)
        if match:
            last, first = match.groups()
            return first, last
        return None, None

    def save_chunks(self, chunks, out_file="cv_chunks.json"):
        out_path = os.path.join("data/", out_file)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)