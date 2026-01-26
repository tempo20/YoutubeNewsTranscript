from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from dotenv import load_dotenv
import os
from youtube_transcript_api.proxies import WebshareProxyConfig
import json
import time
import random

# Load .env from project root (works regardless of working directory)
_project_root = Path(__file__).parent
_env_path = _project_root / '.env'
load_dotenv(dotenv_path=_env_path)

TRANSCRIPT_CACHE_PATH = Path('daily_transcripts.json')
MAX_CACHE_SIZE = 100  # Maximum number of videos to keep in cache

def load_transcript_cache(path):
    if path.exists():
        try:
            content = path.read_text(encoding='utf-8')
            if content.strip():
                return json.loads(content)
            else:
                print("⚠️  Cache file is empty, starting fresh")
                return {}
        except json.JSONDecodeError as e:
            print(f"⚠️  Cache file is corrupted: {e}")
            print("   Starting with fresh cache")
            return {}
    return {}

def save_transcript_cache(path, cache):
    try:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as e:
        print(f"⚠️  Failed to save cache: {e}")

def fetch_transcript_with_backoff(video_id, max_retries=10):
    """
    Fetch transcript with exponential backoff and jitter.
    No proxy - relies on longer delays to avoid rate limits.
    """
    base_delay = 5.0  # Longer initial delay without proxy

    for attempt in range(1, max_retries + 1):
        try:
            # Add random delay before each request (rate limit avoidance)
            jitter = random.uniform(2, 5)
            if attempt > 1:
                time.sleep(jitter)
            ytt_api = YouTubeTranscriptApi(
                proxy_config=WebshareProxyConfig(
                    proxy_username=os.getenv("PROXY_USER"),
                    proxy_password=os.getenv("PROXY_PASS"),
                )
            )
            transcript = ytt_api.fetch(video_id)
            return ' '.join([seg.text for seg in transcript])

        except (TranscriptsDisabled, NoTranscriptFound):
            # These are not rate limits, just unavailable transcripts
            return None

        except Exception as e:
            error_msg = str(e).lower()

            # Check for rate limit indicators
            if any(indicator in error_msg for indicator in ['429', 'too many requests', 'rate limit', 'forbidden', '403']):
                wait_time = base_delay * (2 ** (attempt - 1)) + random.uniform(5, 15)
                print(f"⚠️  Rate limit detected (attempt {attempt}/{max_retries})")
                print(f"   Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
                continue

            # Other errors
            error_type = type(e).__name__
            print(f"Attempt {attempt} failed for {video_id}: {error_type}: {str(e)[:100]}")
            # Skip immediately for unavailable/unplayable videos (no point retrying)
            if any(skip in error_msg or skip in error_type.lower() for skip in
                   ['unavailable', 'unplayable', 'private', 'deleted']):
                print(f"  ⏭️  Skipping (video unavailable)")
                return None
            if attempt == max_retries:
                return None

            # Exponential backoff for other errors
            wait_time = base_delay * (1.5 ** attempt) + random.uniform(1, 3)
            time.sleep(wait_time)

    return None

def attach_transcripts(videos, cache_path=TRANSCRIPT_CACHE_PATH, max_cache_size=MAX_CACHE_SIZE, delay_between_requests=3.0):
    """
    Attach transcripts to videos with aggressive rate limit avoidance.

    Args:
        videos: List of video dictionaries
        cache_path: Path to cache file
        max_cache_size: Maximum number of videos to keep in cache
        delay_between_requests: Base delay between requests in seconds (default: 3.0)
    """
    latest_ids = [v.get('video_id') for v in videos if v.get('video_id')]
    total_videos = len(latest_ids)
    print(f"\n📝 Processing {total_videos} videos for transcripts...\n")
    print(f"⏱️  Using delays to avoid rate limits (no proxy)\n")

    # Load cache
    cache = load_transcript_cache(cache_path)
    old_cache_size = len(cache)

    # Create ordered list: newest videos first
    all_video_ids = latest_ids.copy()

    # Add old cached videos that aren't in the new list
    for old_vid in cache.keys():
        if old_vid not in all_video_ids:
            all_video_ids.append(old_vid)

    # Keep only the newest MAX_CACHE_SIZE videos
    videos_to_keep = all_video_ids[:max_cache_size]

    # Filter cache
    filtered_cache = {vid: cache[vid] for vid in videos_to_keep if vid in cache}
    removed_count = old_cache_size - len(filtered_cache)

    print(f"📦 Cache status: {old_cache_size} total → keeping {len(filtered_cache)} (removed {removed_count} oldest)\n")

    cache = filtered_cache

    success_count = 0
    failed_count = 0
    cached_count = 0
    actual_idx = 0

    for idx, video in enumerate(videos, start=1):
        vid = video.get('video_id')
        if not vid:
            continue

        actual_idx = idx

        # Check cache first
        if vid in cache:
            video['transcript_text'] = cache[vid]
            cached_count += 1
            print(f"[{idx}/{total_videos}] ✓ Cached: {vid} - {video.get('title', 'N/A')[:50]}")
            if cache[vid]:
                print(f"  Preview: {cache[vid][:150]}...\n")
            continue

        # Add delay between requests to avoid rate limits
        delay = delay_between_requests + random.uniform(1, 3)
        print(f"[{idx}/{total_videos}] Fetching: {vid} (waiting {delay:.1f}s)...")
        time.sleep(delay)

        # Fetch transcript
        try:
            transcript_text = fetch_transcript_with_backoff(vid)
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")
            transcript_text = None

        video['transcript_text'] = transcript_text
        cache[vid] = transcript_text

        if transcript_text:
            success_count += 1
            print(f"✓ Success: {video.get('title', 'N/A')[:50]}")
            print(f"  Preview: {transcript_text[:150]}...\n")
        else:
            failed_count += 1
            print(f"✗ Failed/No transcript: {video.get('title', 'N/A')[:50]}\n")

        # Save cache periodically
        if idx % 10 == 0:
            save_transcript_cache(cache_path, cache)
            print(f"  💾 Cache saved at {idx} videos\n")

    # Final save
    save_transcript_cache(cache_path, cache)

    print(f"\n📊 Summary:")
    print(f"  ✓ Successfully fetched: {success_count}")
    print(f"  ✓ From cache: {cached_count}")
    print(f"  ✗ Failed/No transcript: {failed_count}")
    print(f"  Total processed: {actual_idx}/{total_videos}")
    print(f"  📦 Final cache size: {len(cache)}/{max_cache_size}")
    print()

    return videos

def refresh_transcripts_in_dict(videos, cache_path=Path('daily_transcripts.json')):
    """Refresh transcript data from cache file"""
    if not cache_path.exists():
        return videos
    cache = json.loads(cache_path.read_text(encoding='utf-8'))
    updated = 0
    for video in videos:
        vid = video.get('video_id')
        if not vid:
            continue
        cached_value = cache.get(vid)
        if cached_value is not None:
            if video.get('transcript_text') != cached_value:
                video['transcript_text'] = cached_value
                updated += 1
    print(f'Overwrote {updated} transcripts from cache')
    return videos