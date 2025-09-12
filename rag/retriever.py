from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from config import CHROMA_DB_PATH

class RAGRetriever:
    def __init__(self, collection_name="hr_chunks"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vector_store = Chroma(collection_name=collection_name, embedding_function=self.embeddings, persist_directory=CHROMA_DB_PATH)
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        self.llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite")

        template = """
        Tu es un assistant RH. Utilise uniquement le contexte fourni ci-dessous pour répondre
        à la question de l'utilisateur. Si tu ne sais pas, dis que tu ne sais pas.
        Inclue toujours les sources (métadonnées) utilisées.

        Contexte : {context}

        Question : {question}

        Réponse :
        """
        self.qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    @staticmethod
    def build_context_with_sources(docs):
        enriched_chunks = [
            f"[Source: {doc.metadata.get('source', 'inconnu')} | Chunk ID: {doc.metadata.get('chunk_id', 'inconnu')}] {doc.page_content}"
            for doc in docs
        ]
        return "\n\n".join(enriched_chunks)

    def query(self, question):
        retrieved_docs = self.retriever.get_relevant_documents(question)
        context = self.build_context_with_sources(retrieved_docs)
        final_prompt = self.qa_prompt.format(context=context, question=question)
        result = self.llm.invoke(final_prompt)
        return result, context
