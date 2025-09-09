import os

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


######################## Charger les variables d'environnement

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

######################## Dossiers
persist_directory = "/home/jip.wulffele@Digital-Grenoble.local/Documents/15_LLM/project_llm/CVthèque/chroma_db_one_cv"
collection_chroma = "cv_collection"

######################## Hyperparameters
MODEL_EMBEDDINGS = "models/embedding-001"
LAMBDA_MULT = 0.2

query = "Qui aime joue des jeux vidéo en équipe?"
query = "Qui est Ingénieur Agronome"
query = "Est ce que Jip parle Anglais?"
K = 3

######################## Functions

def search_cvs(query, K=K, LAMBDA_MULT=LAMBDA_MULT, last_name=None,):
    embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDINGS)

    vectorstore = Chroma(
        collection_name=collection_chroma,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )


    if last_name:
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # diverse and relevant results
            search_kwargs={
                'k': K, 
                'lambda_mult': LAMBDA_MULT,
                'filter': {'last_name':last_name}
                }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # diverse and relevant results
            search_kwargs={
                'k': K, 
                'lambda_mult': LAMBDA_MULT,
                }
        )

    results = retriever.invoke(query)

    return results

def main(query):
    results = search_cvs(query)
    return results

######################## Main
if __name__ == "__main__":
    results = main(query)

    for r in results:
        print()
        print("#####################################################################################################################")
        print()
        print(f"{r.metadata['source']} → {r.page_content[:200]}...")
    