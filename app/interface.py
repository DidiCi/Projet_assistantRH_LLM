import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
    
import streamlit as st
from rag.retriever import RAGRetriever 

st.set_page_config(page_title="Assistant RH RAG", page_icon="💼")

# Custom styled title
st.markdown(
    "<h1 style='font-size:36px; color:#2E86C1;'>💼 Assistant RH - Chat RAG avec Gemini</h1>",
    unsafe_allow_html=True
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize RAG pipeline only once
if "rag_pipeline" not in st.session_state:
    st.session_state["rag_pipeline"] = RAGRetriever()

# User input
user_query = st.chat_input("Posez votre question ici...")

if user_query:
    # Run RAG pipeline
    pipeline = st.session_state["rag_pipeline"]
    answer, retrieved_docs = pipeline.query(user_query)

    # Save user and assistant messages to history
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.session_state["messages"].append({"role": "assistant", "content": answer})

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show context for last answer
if st.session_state["messages"]:
    with st.expander("🔍 Contexte utilisé pour la dernière réponse"):
        for doc in retrieved_docs:
            st.markdown(f"**Source:** {doc.metadata.get('source', 'inconnu')}")
            st.markdown(f"**Chunk:** {doc.page_content}")
            st.markdown("---")
