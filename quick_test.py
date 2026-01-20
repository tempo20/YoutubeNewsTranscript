"""
Quick test script for per-entity sentiment analysis.
Run this for a fast test on a single video.
"""

import json
from pathlib import Path
from get_entity_mentions import analyze_video_entities_split, debug_video_entities

# Load transcript cache
transcript_path = Path("Testing/daily_transcripts.json")
with open(transcript_path, 'r', encoding='utf-8') as f:
    transcript_cache = json.load(f)

# Get first video with valid transcript
if transcript_cache:
    for video_id, transcript_text in transcript_cache.items():
        if transcript_text:  # Skip None or empty transcripts
            # Create test video
            # Use transcript_text directly (no summary needed for per-entity sentiment)
            test_video = {
                'video_id': video_id,
                'title': transcript_text[:100] + "..." if len(transcript_text) > 100 else transcript_text,
                'transcript_text': transcript_text,
                # transcript_summary not needed - analyze_video_entities_split will use transcript_text
            }
            
            print(f"Testing video: {video_id}")
            print(f"Transcript length: {len(transcript_text)} chars\n")
            
            # Run debug analysis
            debug_video_entities(test_video)
            break
    else:
        print("No valid transcripts found in cache (all are None or empty)")
else:
    print("No transcripts found in cache")
