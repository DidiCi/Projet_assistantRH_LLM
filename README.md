# Projet_assistantRH_LLM

This project was developed in the context of an **LLM training module**, as a practical exercise in building an assistant tool for Human Resources (RH) using **RAG (Retrieval-Augmented Generation)** techniques.

It provides a pipeline to:
- Build a RAG from HR-related data,
- Query the assistant with natural language,
- Interact via a simple interface,
- Evaluate the quality of answers against a dataset.

---

## 🚀 Installation

The project uses **[uv](https://github.com/astral-sh/uv)** for environment and dependency management.

1. **Clone the repository** and switch to the organized branch:
   ```bash
   git clone https://github.com/DidiCi/Projet_assistantRH_LLM.git
   cd Projet_assistantRH_LLM
   ```

2. **Install the environment with uv**:
   ```bash
   uv sync
   ```

3. **Set up configuration**:
   - Obtain a **Google API key** and save it in a `.env` file at the project root:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```
   - Input/output folders and other options can be configured in [`rag/config.py`](rag/config.py).

4. **Prepare the data**:
   - Place your CV files (the documents in PDF format to be analyzed) in:
     ```
     data/raw/
     ```

---

## 🧠 Usage

### 1. Build and test the RAG
Create the RAG:

```bash
uv run python rag/main.py
```

You can also ask a direct question when running it:

```bash
uv run python rag/main.py --question "Qui parle italien?"
```

### 2. Run the interface
Once the RAG is created, launch the Streamlit interface:

```bash
uv run streamlit run app/interface.py
```

This provides a user-friendly way to interact with the assistant.

---

## 📊 Evaluation

The project includes tools to evaluate the RAG’s answers.

1. Define your test set by editing:
   ```
   evaluation/evaluation_dataset.json
   ```
   Add **questions and their expected answers**.

2. Run the evaluation pipeline:

   ```bash
   uv run python evaluation/evaluation_llm.py
   uv run python evaluation/evaluation_score.py
   ```

This will generate scores and metrics about the assistant’s accuracy and relevance.

---

## 📂 Project Structure

| Path | Description |
|------|-------------|
| `app/` | Streamlit interface for interacting with the assistant. |
| `rag/` | Core RAG implementation (retrieval, embeddings, pipeline). |
| `rag/config.py` | Configuration file for input/output folders and settings. |
| `data/raw/` | Folder where input CVs must be placed. |
| `evaluation/` | Scripts and datasets for evaluating RAG answers. |
| `evaluation/evaluation_dataset.json` | JSON dataset of questions & answers for evaluation. |
| `.env` | Must contain the Google API key. |
| `pyproject.toml`, `uv.lock` | Project dependencies managed by `uv`. |

---

## 🔧 Requirements

- [uv](https://github.com/astral-sh/uv)  
- Python (version specified in `.python-version`)  
- Google API key  

Dependencies are automatically installed via `uv sync`.

---

## 📌 Context

This repository was created as part of an **LLM formation module**, to practice:
- Using RAG for domain-specific assistants,
- Managing configurations and pipelines,
- Evaluating model performance systematically,
- Building a minimal interactive application.

---

