
from dotenv import load_dotenv
from retrieve_vids import fetch_youtube_videos_with_api, save_to_json
from retrieve_transcripts import attach_transcripts, refresh_transcripts_in_dict
from get_sentiments import get_all_sentiments
from get_entity_mentions import aggregate_youtube_entities

def main():
    load_dotenv()
    videos = fetch_youtube_videos_with_api()
    videos = attach_transcripts(videos)
    videos = refresh_transcripts_in_dict(videos)
    save_to_json(videos, "no_sentiment_vids.json")
    videos = get_all_sentiments(videos)
    save_to_json(videos, "youtube_analysis.json")
    result = aggregate_youtube_entities(videos)
    save_to_json(result, "entity_mentions.json")
    print(result)
if __name__ == "__main__":
    main()
