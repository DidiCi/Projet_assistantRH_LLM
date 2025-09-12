import json


class EvaluationScorer:
    def __init__(self, results_path="evaluation/evaluation_results.json"):
        self.results_path = results_path
        self.results = self.load_results()

    def load_results(self):
        with open(self.results_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def clean_output(evaluation_str):
        texte_nettoye = evaluation_str.strip()
        for marker in ["```json", "```", '"""']:
            if texte_nettoye.startswith(marker):
                texte_nettoye = texte_nettoye[len(marker):].strip()
            if texte_nettoye.endswith(marker):
                texte_nettoye = texte_nettoye[:-len(marker)].strip()
        return json.loads(texte_nettoye)

    @staticmethod
    def calculate_score(evaluation):
        if isinstance(evaluation, str):
            evaluation = json.loads(evaluation)
        criteria = ["exactitude", "completude", "ton"]
        total = sum(1 if evaluation.get(c, True) else 0 for c in criteria)
        return total / len(criteria)

    def score_results(self):
        for item in self.results:
            item["score"] = self.calculate_score(self.clean_output(item["evaluation"]))
        return self.results

    def print_scores(self):
        print("📊 Scores per answer:")
        for item in self.results:
            print(f"- Question: {item.get('query', 'N/A')}, Score: {item['score']}")
        avg_score = sum(item["score"] for item in self.results) / len(self.results)
        print(f"\n⭐ Average score across all answers: {avg_score:.2f}")
        return avg_score


if __name__ == "__main__":
    scorer = EvaluationScorer()
    scorer.score_results()
    scorer.print_scores()
