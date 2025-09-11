import streamlit as st
from config_streamlit import AppConfig
from extraction_module import CVExtractor
from vectorstore_module import VectorStoreManager
from rag_module import RAGPipeline
from evaluation_module import Evaluator

# -------------------------------
# 1. App Config & Initialization
# -------------------------------
st.set_page_config(page_title="Assistant RH", layout="wide")
st.title("📄 Assistant RH - CV RAG")

# -------------------------------
# 2. Sidebar: Mistral key
# -------------------------------
st.sidebar.header("⚙️ Paramètres Mistral")
mistral_key = st.sidebar.text_input(
    "Mistral API Key",
    type="password",
    help="Entrez votre clé API Mistral ici."
)
if not mistral_key:
    st.warning("Veuillez entrer votre clé API Mistral pour continuer.")
    st.stop() 
if "mistral_key" not in st.session_state:
    st.session_state["mistral_key"] = mistral_key

# -------------------------------
# 2. Sidebar: General parameters
# -------------------------------
st.sidebar.header("⚙️ Paramètres")
filter_lastname = st.sidebar.text_input("Filtrer par nom de famille (optionnel)")
evaluate_answers = st.sidebar.checkbox("Évaluer la réponse", value=True)

st.sidebar.header("⚙️ Paramètres de la base de données")
db_mode = st.sidebar.radio(
    "Mode de la base de données",
    ("Connecter à une base existante", "Créer une nouvelle base")
)

@st.cache_resource
def load_pipeline(db_mode, mistral_key):
    if not mistral_key:
        st.error("La clé Mistral est requise pour initialiser les embeddings !")
        st.stop()

    config = AppConfig()
    create_new = db_mode == "Créer une nouvelle base"

    extractor = CVExtractor(config)
    vs = VectorStoreManager(config, create_new=create_new, mistral_key=mistral_key)  # <-- pass key
    rag = RAGPipeline(config, vs, mistral_key=mistral_key)  # <-- pass key
    evaluator = Evaluator(config, mistral_key=mistral_key)  # <-- pass key
    return config, extractor, vs, rag, evaluator

config, extractor, vs, rag, evaluator = load_pipeline(db_mode, st.session_state["mistral_key"])

# -------------------------------
# 3. Upload CVs
# -------------------------------
st.header("📤 Importer des CVs")
uploaded_files = st.file_uploader("Déposez des fichiers PDF", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if db_mode == "Connecter à une base existante":
        st.info("Vous ajoutez des CVs à la base existante.")
    with st.spinner("Extraction en cours..."):
        for file in uploaded_files:
            file_path = f"{config.folder_path}/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        chunks = extractor.extract_chunks()
        extractor.save_chunks(chunks)
        vs.add_chunks(chunks)
    st.success(f"{len(uploaded_files)} CVs ajoutés et indexés ✅")

# -------------------------------
# 4. Pose a Question
# -------------------------------
st.header("❓ Poser une question")
query = st.text_input("Entrez votre question :")

if query:
    with st.spinner("Recherche et génération de réponse..."):
        answer, results = rag.ask(query, last_name=filter_lastname)
    st.subheader("💬 Réponse de l'assistant")
    st.write(answer)

    # Display retrieved CV chunks
    with st.expander("Voir les CVs pertinents"):
        for i, doc in enumerate(results, 1):
            st.markdown(f"**CV {i} - {doc.metadata.get('first_name','')} {doc.metadata.get('last_name','')}**")
            st.text(doc.page_content[:500] + "...")

    # -------------------------------
    # 5. Evaluation (optional)
    # -------------------------------
    if evaluate_answers:
        with st.spinner("Évaluation de la réponse..."):
            faithfulness = evaluator.eval_faithfulness(query, answer, results)
            bias = evaluator.eval_bias(query, answer)

        st.subheader("📊 Évaluations")
        st.json({
            "Faithfulness": faithfulness.content,
            "Bias": bias.content
        })