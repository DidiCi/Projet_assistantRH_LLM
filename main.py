from config import AppConfig
from extraction_module import CVExtractor
from vectorstore_module import VectorStoreManager
from rag_module import RAGPipeline
from evaluation_module import Evaluator

from dotenv import load_dotenv
import os

CREATE_NEW = True

def main():

    # Load API key
    load_dotenv()
    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if not mistral_key:
        raise ValueError("❌ MISTRAL_API_KEY not found in .env")

    # Set up configuration
    config = AppConfig()

    # Extraction
    extractor = CVExtractor(config)
    chunks = extractor.extract_chunks()
    extractor.save_chunks(chunks)

    # Vectorstore
    vs = VectorStoreManager(config, create_new=CREATE_NEW)
    vs.add_chunks(chunks)

    # RAG Query
    rag = RAGPipeline(config, vs)
    query = "Quels candidats ont de l'expérience en Python ?"
    ground_truth = ""

    answer, results = rag.ask(query)

    print("\n--- RAG results ---\n", results)
    print("\n--- Answer ---\n", answer)

    # Evaluation
    evaluator = Evaluator(config)

    faithfulness = evaluator.eval_faithfulness(query, answer, results)
    print("\n--- Faithfulness ---\n", faithfulness.content)
    bias = evaluator.eval_bias(query, answer)
    print("\n--- Bias ---\n", bias.content)
    semantic = evaluator.eval_semantic(query, ground_truth, answer)
    print("\n--- Semantic---\n", semantic.content)
    context = evaluator.eval_context(query, results)
    print("\n--- Context---\n", context.content)

if __name__ == "__main__":
    main()
