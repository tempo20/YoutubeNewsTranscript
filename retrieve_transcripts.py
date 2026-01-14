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

ytt_api = YouTubeTranscriptApi(
    proxy_config=WebshareProxyConfig(
        proxy_username=os.getenv("PROXY_USER"),
        proxy_password=os.getenv("PROXY_PASS"),
    )
)

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

def fetch_transcript_with_backoff(video_id, max_retries=5):
    delay = 2.0 
    consecutive_bans = 0
    max_consecutive_bans = 10
    
    for attempt in range(1, max_retries + 1):
        try:
            # Get API instance (created on first use, when env vars are definitely loaded)
            transcript = ytt_api.fetch(video_id)
            return ' '.join([seg.text for seg in transcript])
        except (TranscriptsDisabled, NoTranscriptFound):
            return None
        except Exception as e:
            error_msg = str(e).lower()
            # Check for ban/rate limit indicators
            if any(indicator in error_msg for indicator in ['429', 'too many requests', 'rate limit', 'forbidden', '403']):
                consecutive_bans += 1
                print(f"⚠️  RATE LIMIT/BAN DETECTED (attempt {attempt}): {type(e).__name__}")
                
                if consecutive_bans >= max_consecutive_bans:
                    print(f"🚫 {max_consecutive_bans} consecutive bans - all IPs likely exhausted")
                    raise Exception("ALL_IPS_BANNED")
                
                # Wait and let Webshare rotate to next IP
                wait_time = 3 + random.random() * 2
                print(f"   Waiting {wait_time:.1f}s for IP rotation, then retrying...")
                time.sleep(wait_time)
                continue
            
            print(f"Attempt {attempt} failed for {video_id}: {type(e).__name__}: {str(e)[:100]}")
            if attempt == max_retries:
                return None
            time.sleep(delay + random.random() * 2)
            delay *= 2
    
    return None

def attach_transcripts(videos, cache_path=TRANSCRIPT_CACHE_PATH, max_cache_size=MAX_CACHE_SIZE):
    latest_ids = [v.get('video_id') for v in videos if v.get('video_id')]
    total_videos = len(latest_ids)
    print(f"\n📝 Processing {total_videos} videos for transcripts...\n")
    print(f"🌐 Using Webshare rotating proxy (10 IPs in pool)\n")

    # Load full cache
    cache = load_transcript_cache(cache_path)
    old_cache_size = len(cache)
    
    # Create ordered list: newest videos first
    # Assuming videos list is already ordered from newest to oldest
    all_video_ids = latest_ids.copy()
    
    # Add old cached videos that aren't in the new list (to maintain order)
    for old_vid in cache.keys():
        if old_vid not in all_video_ids:
            all_video_ids.append(old_vid)
    
    # Keep only the newest MAX_CACHE_SIZE videos
    videos_to_keep = all_video_ids[:max_cache_size]
    
    # Filter cache to only keep videos we want
    filtered_cache = {vid: cache[vid] for vid in videos_to_keep if vid in cache}
    removed_count = old_cache_size - len(filtered_cache)
    
    print(f"📦 Cache status: {old_cache_size} total → keeping {len(filtered_cache)} (removed {removed_count} oldest)\n")
    
    cache = filtered_cache

    success_count = 0
    failed_count = 0
    cached_count = 0
    all_ips_banned = False

    for idx, video in enumerate(videos, start=1):
        vid = video.get('video_id')
        if not vid:
            continue
        
        if vid in cache:
            video['transcript_text'] = cache[vid]
            cached_count += 1
            print(f"[{idx}/{total_videos}] ✓ Cached: {vid} - {video.get('title', 'N/A')[:50]}")
            if cache[vid]:
                print(f"  Preview: {cache[vid][:150]}...\n")
            continue
        
        time.sleep(1.5 + random.random() * 1.5)
        
        try:
            transcript_text = fetch_transcript_with_backoff(vid)
        except Exception as e:
            if "ALL_IPS_BANNED" in str(e):
                all_ips_banned = True
                print(f"\n🚫 ALL IPs BANNED - Stopping transcript fetching and saving cache...")
                video['transcript_text'] = None
                cache[vid] = None
                save_transcript_cache(cache_path, cache)
                break
            transcript_text = None
        
        video['transcript_text'] = transcript_text
        cache[vid] = transcript_text
        
        if transcript_text:
            success_count += 1
            print(f"[{idx}/{total_videos}] ✓ Fetched: {vid} - {video.get('title', 'N/A')[:50]}")
            print(f"  Preview: {transcript_text[:150]}...\n")
        else:
            failed_count += 1
            print(f"[{idx}/{total_videos}] ✗ Failed: {vid} - {video.get('title', 'N/A')[:50]}\n")
        
        if idx % 10 == 0:
            save_transcript_cache(cache_path, cache)
            print(f"  💾 Cache saved at {idx} videos\n")

    # Final save
    save_transcript_cache(cache_path, cache)
    
    print(f"\n📊 Summary:")
    print(f"  ✓ Successfully fetched: {success_count}")
    print(f"  ✓ From cache: {cached_count}")
    print(f"  ✗ Failed/No transcript: {failed_count}")
    print(f"  Total processed: {idx}/{total_videos}")
    print(f"  📦 Final cache size: {len(cache)}/{max_cache_size}")
    if all_ips_banned:
        print(f"  🚫 Stopped early - all rotating IPs exhausted")
    print()
    
    return videos

def refresh_transcripts_in_dict(videos, cache_path=Path('daily_transcripts.json')):
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
    print(f'Overwrote {updated} transcripts from dict')
    return videos