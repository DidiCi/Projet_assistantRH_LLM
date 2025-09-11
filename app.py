import streamlit as st
from rag_function import run_rag 

st.set_page_config(page_title="Assistant RH RAG", page_icon="💼")

# Custom styled title
st.markdown(
    "<h1 style='font-size:36px; color:#2E86C1;'>💼 Assistant RH - Chat RAG avec Gemini</h1>",
    unsafe_allow_html=True
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# User input
user_query = st.chat_input("Posez votre question ici...")

if user_query:
    # Run RAG pipeline
    answer, retrieved_docs = run_rag(user_query)

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
