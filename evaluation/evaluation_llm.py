import sys
import os

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import json
import logging
from datetime import datetime
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from rag.retriever import RAGRetriever  


class LLMEvaluator:
    def __init__(self, dataset_path="evaluation/evaluation_dataset.json", log_dir="evaluation/logs"):
        self.dataset_path = dataset_path
        self.results = []
        self.rag_pipeline = RAGRetriever()  

        # Judge model
        self.llm_judge = GoogleGenerativeAI(model="gemini-2.5-flash")
        self.judge_prompt = PromptTemplate.from_template("""
        Tu es un assistant RH évaluateur. Analyse la réponse fournie par le système RAG par rapport à la question et à la réponse attendue.

        Critères d'évaluation:
        1. Exactitude: La réponse du système est-elle factuellement correcte ? (True/False)
        2. Complétude: La réponse couvre-t-elle tous les points attendus ? (True/False)
        3. Ton: La réponse est-elle appropriée et professionnelle ? (True/False)
        4. Explication: Fournis une brève justification pour les notes attribuées.

        Question: {question}
        Réponse attendue: {expected}
        Réponse du système: {predicted}

        Réponds uniquement en ce format JSON:
        {{
          "exactitude": true/false,
          "completude": true/false,
          "ton": true/false,
          "explication": "..."
        }}
        """)

        # Setup logging
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"{log_dir}/evaluation_{timestamp}.log"
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def evaluate_with_llm(self, query, expected, predicted):
        prompt = self.judge_prompt.format(
            question=query, expected=expected, predicted=predicted
        )
        return self.llm_judge.invoke(prompt)

    def run(self, output_path="evaluation/evaluation_results.json"):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        for test in eval_data:
            query = test["question"]
            expected = test["expected_answer"]

            predicted, _ = self.rag_pipeline.query(query)
            evaluation = self.evaluate_with_llm(query, expected, predicted)

            self.results.append({
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "evaluation": evaluation,
            })

            logging.info(f"Question: {query}")
            logging.info(f"Expected: {expected}")
            logging.info(f"Predicted: {predicted}")
            logging.info(f"Evaluation: {evaluation}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        print(f"✅ Evaluation finished. Results saved to {output_path}")


if __name__ == "__main__":
    evaluator = LLMEvaluator()
    evaluator.run()
