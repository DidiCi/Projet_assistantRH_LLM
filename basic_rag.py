from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.chains import RetrievalQA
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

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# LLM Gemini
# Modeles disponibles : gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
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

# Chaîne RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # combine les chunks ensemble
    chain_type_kwargs={"prompt": qa_prompt},
    return_source_documents=True
)

# Exemple de requête
# query = "Quelles langues parle Ronan ?"
query = "Qui a un doctorat ? "
result = qa_chain.invoke({"query": query})

print("Réponse :", result["result"])

print("\n Chunks utilisés :")
for doc in result["source_documents"]:
    print("-----")
    print("Source:", doc.metadata.get("source", "inconnu"))
    print("Page:", doc.metadata.get("page", "N/A"))
    print("Texte:", doc.page_content, "...\n")