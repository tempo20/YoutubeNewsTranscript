import json
from pathlib import Path

INPUT_FILE = "entity_mentions.json"
OUTPUT_FILE = "market_summary.json"

MIN_MENTIONS = 2


def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def score_entity(e):
    title_mentions = e.get("title_mentions", 0) or 0
    summary_mentions = e.get("summary_mentions", 0) or 0

    title_sent = e.get("avg_title_sentiment")
    summary_sent = e.get("avg_summary_sentiment")

    sentiments = [s for s in [title_sent, summary_sent] if s is not None]
    avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0.0

    relevance = title_mentions * 2 + summary_mentions

    return {
        "name": e["name"],
        "mentions": title_mentions + summary_mentions,
        "relevance": relevance,
        "sentiment": round(avg_sent, 3)
    }


def preprocess_group(group):
    processed = [score_entity(e) for e in group]
    processed = [e for e in processed if e["mentions"] >= MIN_MENTIONS]
    processed.sort(key=lambda x: x["relevance"], reverse=True)
    return processed


def classify_market_bias(sectors):
    bias = {}
    for s in sectors:
        if s["sentiment"] > 0.15:
            bias[s["name"]] = "bullish"
        elif s["sentiment"] < -0.15:
            bias[s["name"]] = "bearish"
        else:
            bias[s["name"]] = "neutral"
    return bias


def preprocess_json():
    data = load_data()

    sectors = preprocess_group(data["sectors"])
    stocks = preprocess_group(data["stocks"])
    companies = preprocess_group(data["companies"])

    macro_entities = companies[:12]  # most relevant macro drivers

    market_bias = classify_market_bias(sectors)

    output = {
        "sectors_ranked": sectors,
        "stocks_ranked": stocks,
        "macro_entities": macro_entities,
        "market_bias": market_bias
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")