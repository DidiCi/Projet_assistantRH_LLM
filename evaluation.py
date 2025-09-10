from llm_answer import rag_pipeline_auto
from llm_answer import rag_pipeline

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

######################## Charger les variables d'environnement
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST")

######################## Dossiers
persist_directory = "/home/jip.wulffele@Digital-Grenoble.local/Documents/15_LLM/project_llm/CVthèque/chroma_db_by_header_mistral"
collection_chroma = "cv_collection"

######################## Hyperparameters
#MODEL_EMBEDDINGS = "models/embedding-001"
MODEL_EMBEDDINGS = "mistral-embed"

LAMBDA_MULT = 0.2
K = 10

#LLM = ChatGoogleGenerativeAI(model="gemini-1.5-flash-lite", temperature=0) # ran out of quota
LLM = ChatMistralAI(model="mistral-tiny", temperature=0)

######################## Functions




######################## Main

if __name__ == "__main__":
    
    faithfulness_prompt = ChatPromptTemplate.from_template("""
        You are an evaluator.

        Query:
        {{query}}

        Retrieved contexts:
        {{contexts}}

        LLM answer:
        {{answer}}

        Task:
        - Judge whether the answer is faithful to the retrieved contexts.
        - Penalize hallucinations (facts not in the contexts) or contradictions.
        - Score from 0.0 (completely hallucinated) to 1.0 (fully faithful).
        - Explain briefly.

        Respond strictly in JSON:
        {{"score": <float>, "explanation": "<string>"}}
        """)
    
    chain = faithfulness_prompt | LLM

    question, reponse, results = rag_pipeline(K, LAMBDA_MULT, MODEL_EMBEDDINGS, LLM, persist_directory, collection_chroma)

    result = chain.invoke({
        "query": question,
        "contexts": results,
        "answer": reponse
    })

    print("\nLLM judge:")
    print(result.content) 