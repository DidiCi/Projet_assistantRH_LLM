import json
import matplotlib.pyplot as plt

def safe_parse(content):
    try:
        return json.loads(content)
    except Exception:
        return {}

def donut_chart(score, label, color, max_score=5, size=(3,3)):
    fig, ax = plt.subplots(figsize=size)
    # Donut values: score vs. remaining
    ax.pie(
        [score, max_score - score],
        radius=1,
        colors=[color, "lightgray"],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor="white")
    )
    # Add text inside the donut
    ax.text(0, 0, f"{score:.1f}/{max_score}", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set(aspect="equal")
    plt.title(label, fontsize=12)
    return fig
    
