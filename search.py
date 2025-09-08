import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

persist_directory = "/home/jip.wulffele@Digital-Grenoble.local/Documents/15_LLM/project_llm/CVthèque/chroma_db"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma(
    collection_name="cv_collection",
    embedding_function=embeddings,
    persist_directory=persist_directory
)

query = "seaborn"
results = vectorstore.similarity_search(query, k=5)

for r in results:
    print()
    print("#####################################################################################################################")
    print()
    print(f"{r.metadata['source']} → {r.page_content[:200]}...")
   