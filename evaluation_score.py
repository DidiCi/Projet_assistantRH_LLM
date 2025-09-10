import json

with open('evaluation_results.json') as f:
    results = json.load(f)

# Function to calculate a single score from an evaluation dict
def calculate_score(evaluation):
    """
    Returns a score between 0 and 1 based on exactitude, completude, and ton.
    """
    # If evaluation is a string, convert to dict
    if isinstance(evaluation, str):
        evaluation = json.loads(evaluation)
    criteria = ["exactitude", "completude", "ton"]
    total = sum(1 if evaluation.get(c, True) else 0 for c in criteria)
    return total / len(criteria)
print(results)
# Calculate score per answer
for item in results:
    print(item["evaluation"])
    print(item["evaluation"]["exactitude"])
    item["score"] = calculate_score(item["evaluation"])

# Print per-answer scores
print("Scores per answer:")
for item in results:
    print(f"Question: {item.get('question', 'N/A')}, Score: {item['score']}")

# Calculate average score
average_score = sum(item["score"] for item in results) / len(results)
print("\nAverage score across all answers:", average_score)
