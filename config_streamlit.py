from dataclasses import dataclass

@dataclass
class AppConfig:
    folder_path: str = "data/CVs"
    persist_directory: str = "data/Chroma_DB_streamlit"
    collection_name: str = "cv_collection"

    chunk_size: int = 800
    chunk_overlap: int = 100
    image_resolution_scale: float = 2.0
    split_by_header: bool = True
    one_chunk_cv: bool = False

    model_embeddings: str = "mistral-embed"
    llm_model: str = "mistral-tiny"
    llm_temperature: float = 0.3

    k: int = 10
    lambda_mult: float = 0.2