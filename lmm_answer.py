from dotenv import load_dotenv

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langfuse.langchain import CallbackHandler


######################## Charger les variables d'environnement
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST")

######################## Dossiers
folder_path = "/home/jip.wulffele@Digital-Grenoble.local/Documents/15_LLM/project_llm/CVthèque/"

persist_directory = "/home/jip.wulffele@Digital-Grenoble.local/Documents/15_LLM/project_llm/CVthèque/chroma_db_by_header_mistral"
collection_chroma = "cv_collection"

######################## Hyperparameters
#MODEL_EMBEDDINGS = "models/embedding-001"
MODEL_EMBEDDINGS = "mistral-embed"

LAMBDA_MULT = 0.2
K = 10

# Modèle léger
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-lite", temperature=0) # ran out of quota
llm = ChatMistralAI(model="mistral-tiny", temperature=0)

# Meta-prompt 
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
        "Tu es un assistant RH. "
        "Tu disposes de CVs convertis en texte. "
        "Chaque extrait de CV contient aussi des métadonnées : prénom (`first_name`), nom (`last_name`) et source du fichier. "
        "Lorsque tu présentes une information, indique clairement à quel candidat elle appartient "
        "(exemple : 'Martin Dupont a 5 ans d'expérience en Python'). "
        "Utilise uniquement les informations fournies dans les CV. "
        "Si tu ne sais pas, réponds que tu ne sais pas. "
        "Sois concis, factuel et professionnel."),
        ("human", 
        "Question : {question}\n\n"
        "CVs pertinents :\n{context}")
    ])

######################## Functions

def search_cvs(query, K=K, LAMBDA_MULT=LAMBDA_MULT, last_name=None,):
    #embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDINGS)
    embeddings = MistralAIEmbeddings(model=MODEL_EMBEDDINGS)
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


def ask_user_for_person():
    answer = input("Souhaitez-vous poser une question concernant une personne en particulier ? (Oui/Non): ").strip().lower()
    if answer in ["oui", "o", "yes", "y"]:
        name = input("Entrez le nom de famille de la personne : ").strip().upper()
        return name
    return None


def answer_query(query, results, callbacks=None):
    context = "\n\n".join(
        f"CV : {doc.metadata.get('first_name', '')} {doc.metadata.get('last_name', '')}\n{doc.page_content}"
        for doc in results
    )
    prompt = RAG_PROMPT.format(question=query, context=context)
    
    response = llm.invoke(prompt, config={"callbacks": callbacks or []})
    return response.content


def rag_pipeline():
    nom_filtre = ask_user_for_person()
    question = input("Posez votre question : ")
    results = search_cvs(question,  K=K, LAMBDA_MULT=LAMBDA_MULT, last_name=nom_filtre)

    langfuse_handler = CallbackHandler()
    reponse = answer_query(question, results, callbacks=[langfuse_handler])
    print("\n--- Réponse ---\n")
    print(reponse)
    print("\n--- RAG results ---\n")
    print(results)


if __name__ == "__main__":
    rag_pipeline()