import json
from rag_function import run_rag
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
import logging
from datetime import datetime

# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"evaluation/logs/evaluation_{timestamp}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Judge model
llm_judge = GoogleGenerativeAI(model="gemini-2.5-flash")

judge_prompt = PromptTemplate.from_template("""
Tu es un assistant RH évaluateur. Analyse la réponse fournie par le système RAG par rapport à la question et à la réponse attendue.

Critères d'évaluation:
1. Exactitude: La réponse du système est-elle factuellement correcte ? (True/False)
2. Complétude: La réponse couvre-t-elle tous les points attendus ? (True/False)
3. Ton: La réponse est-elle appropriée et professionnelle ? (True/False)
4. Explication: Fournis une brève justification pour les notes attribuées.

Question: {question}
Réponse attendue: {expected}
Réponse du système: {predicted}

Réponds uniquement en ce format avec les clés: "exactitude", "completude", "ton", "explication". Exemple:

{{
  "exactitude": True,
  "completude": True,
  "ton": True,
  "explication": "La réponse du système correspond exactement à la réponse attendue et utilise un ton approprié."
}}
""")

def evaluate_with_llm(query, expected, predicted):
    prompt = judge_prompt.format(
        question=query, expected=expected, predicted=predicted
    )
    return llm_judge.invoke(prompt)

# Load evaluation dataset
with open("evaluation/evaluation_dataset.json", "r") as f:
    eval_data = json.load(f)

results = []

for test in eval_data:
    query = test["question"]
    expected = test["expected_answer"]

    predicted, sources = run_rag(query)
    evaluation = evaluate_with_llm(query, expected, predicted)
    results.append({"query": query, " expected": expected, "predicted": predicted, "evaluation": evaluation})

    # Log the process
    logging.info(f"Question: {query}")
    logging.info(f"Expected: {expected}")
    logging.info(f"Predicted: {predicted}")
    logging.info(f"Evaluation: {evaluation}")

# Write results to JSON
with open("evaluation/evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
    

