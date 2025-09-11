from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

class VectorStoreManager:
    def __init__(self, config, persist_directory=None, create_new=False, mistral_key=None):
        self.config = config
        self.embeddings = MistralAIEmbeddings(model=config.model_embeddings,
                                              api_key=mistral_key)

        self.persist_directory = persist_directory or config.persist_directory

        # If create_new = True → we expect to add fresh chunks
        # If False → connect to an existing DB
        self.vectorstore = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        if create_new:
            # If rebuilding, clear old collection
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=config.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

    def add_chunks(self, chunks):
        texts = [c["text"] for c in chunks]
        metadatas = [
            {"source": c["source"], "first_name": c["first_name"], "last_name": c["last_name"]}
            for c in chunks
        ]
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def search(self, query, last_name=None):
        kwargs = {"k": self.config.k, "lambda_mult": self.config.lambda_mult}
        if last_name:
            kwargs["filter"] = {"last_name": last_name}
        retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs=kwargs)
        return retriever.invoke(query)