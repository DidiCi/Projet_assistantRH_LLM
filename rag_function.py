from config import GOOGLE_API_KEY
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Wrapper to build RAG pipeline
def build_rag():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vector_store = Chroma(
        collection_name="hr_chunks",  
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

    llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite") 

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

    return retriever, llm, qa_prompt

# Build context with metadata
def build_context_with_sources(docs):
    enriched_chunks = []
    for doc in docs:
        src = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("chunk_id", "inconnu")
        enriched_chunks.append(f"[Source: {src} | Chunck ID: {page}] {doc.page_content}")
    return "\n\n".join(enriched_chunks)

# Run one query
def run_rag(query):
    retriever, llm, qa_prompt = build_rag()
    retrieved_docs = retriever.get_relevant_documents(query)
    context = build_context_with_sources(retrieved_docs)
    final_prompt = qa_prompt.format(context=context, question=query)
    result = llm.invoke(final_prompt)
    return result, retrieved_docs
