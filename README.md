# 📄 Assistant RH - CV RAG

Un assistant RH interactif basé sur **RAG (Retrieval-Augmented Generation)** permettant d’importer des CVs, de poser des questions, et d’évaluer la qualité des réponses générées grâce aux modèles **Mistral AI**.

## 🚀 Fonctionnalités

- 📤 Import de CVs au format **PDF**
- 🔎 Indexation et recherche via **ChromaDB**
- ❓ Pose de questions en langage naturel
- 🤖 Génération de réponses avec **Mistral**
- 📊 Évaluation automatique des réponses (fiabilité, biais, contexte, sémantique)
- 🎨 Visualisation des scores (texte ou graphiques en donut)

---

## 📂 Structure du projet

```
PROJET_ASSISTANTRH_LLM/
│── app.py                   # Application principale (Streamlit)
│── main.py                  # Point d’entrée alternatif
│── config.py                # Config générale
│── config_streamlit.py      # Config spécifique à l’app
│── evaluation_module.py     # Module d’évaluation des réponses
│── extraction_module.py     # Extraction et segmentation des CVs
│── rag_module.py            # Logique RAG (Recherche + Génération)
│── vectorstore_module.py    # Gestion de la base vectorielle (Chroma)
│── utils.py                 # Fonctions utilitaires (parse, donut chart…)
│── test.ipynb               # Notebook de tests
│── README.md                # Documentation du projet
│── pyproject.toml           # Dépendances gérées par uv
│── .env                     # Variables d’environnement (clé API, chemins…)
│── .gitignore
│── data/                    # ⚠️ Ignoré par git – à créer manuellement
```

---

## 📂 Dossier `data/`

Comme le dossier `data/` est dans le `.gitignore`, vous devez le créer **manuellement** après avoir cloné le projet.  
Voici la structure attendue :

```
data/
├── CVs/                 # Déposer vos CVs PDF ici
├── Chroma_DB/           # Base vectorielle (créée automatiquement)
├── Chroma_DB_streamlit/ # Base utilisée par l’application Streamlit
```

Les fichiers JSON (`cv_chunks.json`) seront générés automatiquement par l’application.

---

## 🛠️ Installation

Ce projet utilise [**uv**](https://docs.astral.sh/uv/guides/projects/) pour la gestion des dépendances.

### 1. Cloner le dépôt

```bash
git clone https://github.com/<VOTRE_UTILISATEUR>/PROJET_ASSISTANTRH_LLM.git
cd PROJET_ASSISTANTRH_LLM
```

### 2. Créer le dossier `data/`

```bash
mkdir -p data/CVs data/Chroma_DB data/Chroma_DB_streamlit
```

### 3. Installer les dépendances

```bash
uv sync
```

### 4. Configurer les variables d’environnement

Créer un fichier `.env` à la racine du projet :

```ini
# .env
MISTRAL_API_KEY="votre_cle_api_mistral"
```

### 5. Lancer l’application

```bash
uv run streamlit run app.py
```

---

## 🔑 Pré-requis

- Python 3.10+
- Compte et clé API Mistral ([docs](https://docs.mistral.ai/))
- Outil `uv` installé :

```bash
pip install uv
```

---

## 📊 Exemple d’utilisation

1. Démarrer l’app avec :

```bash
uv run streamlit run app.py
```

2. Entrer votre **clé API Mistral** dans la barre latérale  
3. Importer un ou plusieurs **CVs en PDF** dans `data/CVs/`  
4. Poser une question du type :  
   > "Quels candidats ont de l’expérience en Python ?"  
5. Consulter la réponse générée et les **évaluations automatiques**

---

## ✅ Prochaines améliorations

- Support multilingue (FR/EN)
- Enrichissement de l’évaluation des biais
- Export des résultats (CSV/JSON)