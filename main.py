from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from project root (works regardless of working directory)
_project_root = Path(__file__).parent
_env_path = _project_root / '.env'
load_dotenv(dotenv_path=_env_path)

from retrieve_vids import fetch_youtube_videos_with_api, save_to_json
from retrieve_transcripts import attach_transcripts, refresh_transcripts_in_dict
from get_sentiments import get_all_sentiments
from get_entity_mentions import aggregate_youtube_entities
from preprocessing_market_entities import preprocess_json

def print_top_mentions(result, top_n=3):

    def merge_entities(group):
        merged = {}

        for e in group:
            name = e["name"]
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "title_mentions": 0,
                    "summary_mentions": 0
                }

            merged[name]["title_mentions"] += e.get("title_mentions", 0)
            merged[name]["summary_mentions"] += e.get("summary_mentions", 0)

        return list(merged.values())

    def top_entities(group):
        merged = merge_entities(group)
        return sorted(
            merged,
            key=lambda x: x["title_mentions"] + x["summary_mentions"],
            reverse=True
        )[:top_n]

    print("\nTop Stocks:")
    for e in top_entities(result["stocks"]):
        total = e["title_mentions"] + e["summary_mentions"]
        print(f"  {e['name']} – {total} mentions")

    print("\nTop Companies:")
    for e in top_entities(result["companies"]):
        total = e["title_mentions"] + e["summary_mentions"]
        print(f"  {e['name']} – {total} mentions")

    print("\nTop Sectors:")
    for e in top_entities(result["sectors"]):
        total = e["title_mentions"] + e["summary_mentions"]
        print(f"  {e['name']} – {total} mentions")

def main():
    videos = fetch_youtube_videos_with_api()
    # videos = attach_transcripts(videos)
    # videos = refresh_transcripts_in_dict(videos)
    # save_to_json(videos, "no_sentiment_vids.json")
    videos = get_all_sentiments(videos, titles_only=True)
    save_to_json(videos, "youtube_analysis.json")
    result = aggregate_youtube_entities(videos)
    save_to_json(result, "entity_mentions.json")
    print_top_mentions(result)
    preprocess_json()

if __name__ == "__main__":
    main()
