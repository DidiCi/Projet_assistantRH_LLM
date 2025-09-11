import streamlit as st
from config_streamlit import AppConfig
from extraction_module import CVExtractor
from vectorstore_module import VectorStoreManager
from rag_module import RAGPipeline
from evaluation_module import Evaluator

import pandas as pd
import matplotlib.pyplot as plt
from utils import safe_parse

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
if evaluate_answers:
    ground_truth = st.text_input("Réponse attendue (optionnel pour évaluation) :")

if st.button("🚀 Lancer la recherche et génération"):
    if not query.strip():
        st.warning("Veuillez entrer une question avant de lancer la recherche.")
    else:
        with st.spinner("Recherche et génération de réponse..."):
            answer, results = rag.ask(query, last_name=filter_lastname)
    
    # Save results to session_state
    st.session_state.answer = answer
    st.session_state.results = results
    # Clear old evaluations so new ones can be recalculated
    if "eval_results" in st.session_state:
        del st.session_state.eval_results

    # -------------------------------
    # Display answer if available
    # -------------------------------
    if "answer" in st.session_state and "results" in st.session_state:
        st.subheader("💬 Réponse de l'assistant")
        st.write(st.session_state.answer)

        with st.expander("Voir les CVs pertinents"):
            for i, doc in enumerate(st.session_state.results, 1):
                st.markdown(f"**CV {i} - {doc.metadata.get('first_name','')} {doc.metadata.get('last_name','')}**")
                st.text(doc.page_content[:500] + "...")

    # -------------------------------
    # 5. Evaluation (optional)
    # -------------------------------
    if evaluate_answers and "eval_results" not in st.session_state:
        with st.spinner("Évaluation de la réponse..."):
            faithfulness = evaluator.eval_faithfulness(query, answer, results)
            bias = evaluator.eval_bias(query, answer)
            context = evaluator.eval_context(query, results)
            semantic = evaluator.eval_semantic(query, ground_truth, answer) if ground_truth else None

        # Extract scores safely
        faithfulness_json = safe_parse(faithfulness.content)
        bias_json = safe_parse(bias.content)
        context_json = safe_parse(context.content)
        semantic_json = safe_parse(semantic.content) if semantic else {}

        scores = {
            "Faithfulness": faithfulness_json.get("score", 0),
            "Bias": bias_json.get("answer_score", 0),
            "Context": (
                context_json.get("precision_score", 0) +
                context_json.get("relevance_score", 0)
            ) / 2,
        }
        if semantic_json:
            scores["Semantic"] = semantic_json.get("score", 0)

        # Compute overall average
        scores["Overall"] = sum(scores.values()) / len(scores)

        # Save everything to session_state so we don’t recompute
        st.session_state.eval_results = {
            "faithfulness": faithfulness,
            "bias": bias,
            "context": context,
            "semantic": semantic,
            "scores": scores,
        }

    # -------------------------------
    # Display results (toggle freely)
    # -------------------------------
    if evaluate_answers and "eval_results" in st.session_state:
        eval_data = st.session_state.eval_results
        scores = eval_data["scores"]

        st.subheader("📊 Évaluations")
        display_as_graph = st.radio("Mode d’affichage", ["Texte", "Graphique"], horizontal=True)

        if display_as_graph == "Graphique":
            df = pd.DataFrame.from_dict(scores, orient="index", columns=["Score"])

            fig, ax = plt.subplots()
            df["Score"].plot(kind="bar", ax=ax, color="skyblue")
            ax.set_ylim(0, 5)  # scores on 0-5 scale
            ax.set_ylabel("Score")
            ax.set_xlabel("Metric")
            ax.set_title("Évaluations des réponses")

            # Annotate bars
            for i, v in enumerate(df["Score"]):
                ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontweight="bold")

            st.pyplot(fig)

            # Global score
            st.metric("Score global moyen", f"{scores['Overall']:.2f}/5")

        else:  # JSON mode
            if eval_data["semantic"]:
                st.json({
                    "Faithfulness": eval_data["faithfulness"].content,
                    "Bias": eval_data["bias"].content,
                    "Semantic": eval_data["semantic"].content,
                    "Context": eval_data["context"].content,
                })
            else:
                st.json({
                    "Faithfulness": eval_data["faithfulness"].content,
                    "Bias": eval_data["bias"].content,
                    "Context": eval_data["context"].content,
                })