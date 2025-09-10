from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os

# Set API key for Google Gemini
from config import GOOGLE_API_KEY

# Initialize the same embeddings as before
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load the existing Chroma vector store
vector_store = Chroma(
    collection_name="hr_chunks",  
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

# LLM Gemini
llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite") 

# Prompt
template = """
Tu es un assistant RH. Utilise uniquement le contexte fourni ci-dessous pour répondre
à la question de l'utilisateur. Si tu ne sais pas, dis que tu ne sais pas.
Inclue toujours les sources (métadonnées) utilisées.

Contexte :
{context}

Question : {question}

Réponse :
"""

qa_prompt = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# Wrapper custom pour injecter les métadonnées dans le contexte
def build_context_with_sources(docs):
    """Construit un contexte enrichi avec la source ou autres métadonnées"""
    enriched_chunks = []
    for doc in docs:
        src = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("chunk_id", "inconnu")
        enriched_chunks.append(f"[Source: {src} | Chunck ID: {page}] {doc.page_content}")
    return "\n\n".join(enriched_chunks)

# Exemple de requête
query = "Si tu devais embaucher quelqu'un, tu prendrais qui et pourquoi ?"

# On recupère les chuncks pertinents
retrieved_docs = retriever.get_relevant_documents(query)

# On utilise notre wrapper pour construire le contexte
context = build_context_with_sources(retrieved_docs)

# Exécuter l'appel au LLM avec notre prompt custom
final_prompt = qa_prompt.format(context=context, question=query)
result = llm.invoke(final_prompt)

print("Réponse :", result)
print("\n\n")

# Debug : imprimer le contexte qui sera envoyé
print("===== CONTEXTE ENVOYÉ À LLM =====")
print(context)
print("=================================\n")

