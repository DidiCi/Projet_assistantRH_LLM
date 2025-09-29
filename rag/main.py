import argparse
from extractor import PDFExtractor
from ingestor import VectorStoreIngestor
from retriever import RAGRetriever

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF and optionally run a RAG query.")
    parser.add_argument("--question", type=str, help="Question to query the vector database")
    args = parser.parse_args()

    # Step 1: Extract PDF chunks
    extractor = PDFExtractor()
    chunk_file = extractor.extract_chunks()

    # Step 2: Ingest into Chroma
    ingestor = VectorStoreIngestor(chunk_file)
    ingestor.ingest()

    # Step 3: Conditional RAG query
    if args.question:
        retriever = RAGRetriever()
        answer, context = retriever.query(args.question)
        print("Réponse :", answer)
        print("\n===== CONTEXTE ENVOYÉ =====\n", context)
    else:
        print("Vector database created. No question provided, skipping RAG query.")

if __name__ == "__main__":
    main()
