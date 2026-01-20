"""
Test script for per-entity sentiment analysis using daily_transcripts.json

This script loads transcripts from daily_transcripts.json and tests the
per-entity sentiment analysis functionality.
"""

import json
from pathlib import Path
from get_entity_mentions import (
    aggregate_youtube_entities,
    analyze_video_entities_split,
    debug_video_entities
)

def load_transcript_cache(path):
    """Load transcript cache from JSON file."""
    if not path.exists():
        print(f"Error: File {path} not found")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_cache_to_videos(transcript_cache, max_videos=None):
    """
    Convert transcript cache (dict of video_id -> transcript_text) to video list format.
    
    Args:
        transcript_cache: Dictionary mapping video_id to transcript_text
        max_videos: Maximum number of videos to process (None for all)
    
    Returns:
        List of video dictionaries
    """
    videos = []
    items = list(transcript_cache.items())
    
    if max_videos:
        items = items[:max_videos]
    
    for video_id, transcript_text in items:
        # Skip if transcript_text is None or empty
        if not transcript_text:
            continue
        
        # Create minimal video object
        # Note: We don't have titles, so we'll use first 100 chars of transcript as placeholder
        title = transcript_text[:100] + "..." if len(transcript_text) > 100 else transcript_text
        
        videos.append({
            'video_id': video_id,
            'title': title,
            'transcript_text': transcript_text,
            # transcript_summary not needed - analyze_video_entities_split will use transcript_text
        })
    
    return videos

def test_single_video(video):
    """Test per-entity sentiment analysis on a single video."""
    print("=" * 80)
    print(f"Testing Video ID: {video.get('video_id', 'N/A')}")
    print(f"Title: {video.get('title', 'N/A')[:100]}")
    print(f"Transcript length: {len(video.get('transcript_text', ''))} chars")
    print("=" * 80)
    print()
    
    # Run analysis
    parts = analyze_video_entities_split(video)
    
    # Display results
    for part_name, part_data in parts.items():
        print(f"\n[{part_name.upper()}]")
        
        # Count entities
        tickers = list(part_data.get('tickers', {}).keys())
        companies = list(part_data.get('companies', {}).keys())
        sectors = list(part_data.get('sectors', {}).keys())
        
        print(f"  Found {len(tickers)} tickers, {len(companies)} companies, {len(sectors)} sectors")
        
        # Show entities with sentiment
        for entity_type in ['tickers', 'companies', 'sectors']:
            entities = part_data.get(entity_type, {})
            if entities:
                print(f"\n  {entity_type.capitalize()} with sentiment:")
                for entity_name, sentiment_list in list(entities.items())[:5]:  # Show first 5
                    if sentiment_list:
                        avg_sent = sum(s for _, s, _, _ in sentiment_list) / len(sentiment_list)
                        print(f"    {entity_name}: {avg_sent:.3f} ({len(sentiment_list)} mention(s))")
                    else:
                        print(f"    {entity_name}: No sentiment data")
    
    print("\n" + "=" * 80 + "\n")

def test_aggregation(videos, max_videos=10):
    """Test full aggregation on multiple videos."""
    print(f"\n{'=' * 80}")
    print(f"Testing aggregation on {min(len(videos), max_videos)} videos")
    print(f"{'=' * 80}\n")
    
    # Limit videos for testing
    test_videos = videos[:max_videos] if len(videos) > max_videos else videos
    
    # Run aggregation
    result = aggregate_youtube_entities(test_videos)
    
    # Display summary
    print(f"Results:")
    print(f"  Stocks: {len(result.get('stocks', []))}")
    print(f"  Companies: {len(result.get('companies', []))}")
    print(f"  Sectors: {len(result.get('sectors', []))}")
    
    # Helper function to format sentiment
    def format_sentiment(sent):
        if sent is None:
            return 'N/A'
        return f"{sent:.4f}"
    
    # Show top entities
    print(f"\n{'=' * 80}")
    print("Top 10 Stocks by Mentions:")
    print(f"{'=' * 80}")
    for i, stock in enumerate(result['stocks'][:10], 1):
        total_mentions = stock['title_mentions'] + stock['summary_mentions']
        title_sent = format_sentiment(stock.get('avg_title_sentiment'))
        summary_sent = format_sentiment(stock.get('avg_summary_sentiment'))
        print(f"{i:2d}. {stock['name']:8s} | Mentions: {total_mentions:3d} | "
              f"Title: {title_sent:>7s} | Summary: {summary_sent:>7s}")
    
    print(f"\n{'=' * 80}")
    print("Top 5 Companies by Mentions:")
    print(f"{'=' * 80}")
    for i, company in enumerate(result['companies'][:5], 1):
        total_mentions = company['title_mentions'] + company['summary_mentions']
        title_sent = format_sentiment(company.get('avg_title_sentiment'))
        summary_sent = format_sentiment(company.get('avg_summary_sentiment'))
        print(f"{i:2d}. {company['name']:30s} | Mentions: {total_mentions:3d} | "
              f"Title: {title_sent:>7s} | Summary: {summary_sent:>7s}")
    
    print(f"\n{'=' * 80}")
    print("Top 5 Sectors by Mentions:")
    print(f"{'=' * 80}")
    for i, sector in enumerate(result['sectors'][:5], 1):
        total_mentions = sector['title_mentions'] + sector['summary_mentions']
        title_sent = format_sentiment(sector.get('avg_title_sentiment'))
        summary_sent = format_sentiment(sector.get('avg_summary_sentiment'))
        print(f"{i:2d}. {sector['name']:20s} | Mentions: {total_mentions:3d} | "
              f"Title: {title_sent:>7s} | Summary: {summary_sent:>7s}")
    
    # Save results
    output_path = Path("Testing/test_entity_mentions.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

def main():
    """Main test function."""
    # Path to transcript cache
    transcript_path = Path("Testing/daily_transcripts.json")
    
    print("Loading transcript cache...")
    transcript_cache = load_transcript_cache(transcript_path)
    
    if not transcript_cache:
        print("No transcripts found in cache file")
        return
    
    print(f"Loaded {len(transcript_cache)} transcripts from cache")
    
    # Convert to video format
    print("\nConverting to video format...")
    videos = convert_cache_to_videos(transcript_cache, max_videos=None)
    print(f"Converted {len(videos)} videos")
    
    if not videos:
        print("No videos to process")
        return
    
    # Test 1: Single video analysis
    print("\n" + "=" * 80)
    print("TEST 1: Single Video Analysis")
    print("=" * 80)
    if videos:
        test_single_video(videos[0])
    
    # Test 2: Full aggregation (on subset for speed)
    print("\n" + "=" * 80)
    print("TEST 2: Full Aggregation")
    print("=" * 80)
    test_aggregation(videos, max_videos=10)
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
