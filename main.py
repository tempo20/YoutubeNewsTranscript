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
from prep_file import convert_entity_mentions_to_text, filter_result
import argparse
from datetime import datetime, timedelta, time

def filter_by_date_range(videos, chosen_date_str):
    chosen_date = datetime.strptime(chosen_date_str, "%Y-%m-%d")

    start_date = chosen_date - timedelta(days=7)
    end_date = datetime.combine(chosen_date.date(), time.max)  # end of day

    filtered = []

    for v in videos:
        published_str = v.get("published") or v.get("published_date")
        if not published_str:
            continue

        published_dt = datetime.fromisoformat(published_str.replace("Z", ""))

        if start_date <= published_dt <= end_date:
            filtered.append(v)

    return filtered


# def main():
#     videos = fetch_youtube_videos_with_api(max_results=560)
#     videos = attach_transcripts(videos)
#     videos = refresh_transcripts_in_dict(videos)
#     save_to_json(videos, "no_sentiment_vids.json")
#     # videos = get_all_sentiments(videos)
#     # save_to_json(videos, "youtube_analysis.json")
#     result = aggregate_youtube_entities(videos)
#     save_to_json(result, "entity_mentions.json")
#     print_top_mentions(result)
#     # preprocess_json()
#     convert_entity_mentions_to_text(result, "entity_mentions.txt")

# if __name__ == "__main__":
#     main()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-transcripts', action='store_true', 
                       help='Skip fetching transcripts (only use titles)')
    parser.add_argument('--skip-sentiment', action='store_true', 
                       help='Skip generating sentiment')
    parser.add_argument('--titles-only', action='store_true',
                       help='Only analyze sentiment for titles (skip transcript summaries)')
    parser.add_argument('--max-results', type=int, default=560,
                       help='Maximum number of videos to fetch (default: 560)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for filtering (YYYY-MM-DD format). If not provided, uses today.')
    args = parser.parse_args()
    
    videos = fetch_youtube_videos_with_api(max_results=args.max_results)
    print(f"Fetched {len(videos)} videos from API")
    
    # Filter by date range
    if args.end_date:
        end_date_str = args.end_date
    else:
        # Default to today if not provided (format as YYYY-MM-DD)
        end_date_str = datetime.now().strftime("%Y-%m-%d")
    
    videos = filter_by_date_range(videos, end_date_str)
    print(f"Filtered to {len(videos)} videos in date range")
    
    if len(videos) == 0:
        print("WARNING: No videos after date filtering! Check your date range.")
        print(f"Date range: {end_date_str} (and 7 days prior)")
        return
    
    if not args.skip_transcripts:
        videos = attach_transcripts(videos)
        videos = refresh_transcripts_in_dict(videos)
        save_to_json(videos, "before_aggregation.json")
    else:
        save_to_json(videos, "before_aggregation.json")
    
    print(f"Videos after transcript processing: {len(videos)}")
    if videos:
        print(f"Sample video title: {videos[0].get('title', 'N/A')[:100]}")

    if not args.skip_sentiment:
        if args.titles_only:
            videos = get_all_sentiments(videos, titles_only=True)
        else:
            videos = get_all_sentiments(videos)
        save_to_json(videos, "after_sentiment.json")
    else:
        save_to_json(videos, "after_sentiment.json")
    
    result = aggregate_youtube_entities(videos)
    print(f"Before filtering - Stocks: {len(result.get('stocks', []))}, Companies: {len(result.get('companies', []))}, Sectors: {len(result.get('sectors', []))}")
    
    # Save unfiltered result first
    save_to_json(result, "entity_mentions.json")
    
    # Apply filter
    filtered_result = filter_result(result)
    print(f"After filtering - Stocks: {len(filtered_result.get('stocks', []))}, Companies: {len(filtered_result.get('companies', []))}, Sectors: {len(filtered_result.get('sectors', []))}")
    
    # Save filtered result back to JSON
    save_to_json(filtered_result, "entity_mentions.json")
    convert_entity_mentions_to_text("entity_mentions.json", "entity_mentions.txt")

if __name__ == "__main__":
    main()