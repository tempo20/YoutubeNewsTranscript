import ast
import json
import os
from pathlib import Path

from transformers import pipeline
# import io
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import logging
logging.set_verbosity_error()

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

SENTIMENT_CACHE_PATH = Path("youtube_analysis.json")
MAX_SENTIMENT_CACHE_SIZE = 100


def load_sentiment_cache(path: Path):
    default_format = "videos" if path and path.name == "youtube_analysis.json" else "map"
    if not path or not path.exists():
        return {}, default_format
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return {}, default_format
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(content)
        except Exception:
            print("Warning: sentiment cache is unreadable, starting fresh.")
            return {}, default_format

    if isinstance(data, dict):
        if "videos" in data and isinstance(data["videos"], list):
            data = data["videos"]
        else:
            cache = {}
            for vid, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                cache[vid] = {
                    "title_sentiment": payload.get("title_sentiment"),
                    "transcript_summary": payload.get("transcript_summary"),
                    "transcript_sentiment": payload.get("transcript_sentiment"),
                }
            return cache, "map"

    if isinstance(data, list):
        cache = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            vid = item.get("video_id")
            if not vid:
                continue
            cache[vid] = {
                "title_sentiment": item.get("title_sentiment"),
                "transcript_summary": item.get("transcript_summary"),
                "transcript_sentiment": item.get("transcript_sentiment"),
            }
        return cache, "videos"

    return {}, default_format


def save_sentiment_cache(path: Path, cache, videos=None, cache_format="map"):
    if not path:
        return
    if cache_format == "videos" and videos is not None:
        path.write_text(json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def prune_sentiment_cache(cache, latest_ids, max_cache_size):
    if not cache or not latest_ids:
        return {}
    if max_cache_size:
        latest_ids = latest_ids[:max_cache_size]
    return {vid: cache[vid] for vid in latest_ids if vid in cache}


def update_sentiment_cache(cache, video):
    vid = video.get("video_id")
    if not vid:
        return
    cache[vid] = {
        "title_sentiment": video.get("title_sentiment"),
        "transcript_summary": video.get("transcript_summary"),
        "transcript_sentiment": video.get("transcript_sentiment"),
    }


def chunk_text_words(text, chunk_words=500):
    words = text.split()
    return [
        " ".join(words[i:i+chunk_words])
        for i in range(0, len(words), chunk_words)
    ]

def summarize_long_text(text):
    chunks = chunk_text_words(text, chunk_words=500)

    partial_summaries = []
    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=120,
            min_length=40,
            do_sample=False,
            truncation=True
        )
        partial_summaries.append(result[0]["summary_text"])

    combined = " ".join(partial_summaries)

    final = summarizer(
        combined,
        max_length=180,
        min_length=60,
        do_sample=False,
        truncation=True
    )

    return final[0]["summary_text"]


def analyze_video_sentiment(video, debug=False):
    """Analyze sentiment of title and transcript separately"""
    
    # Title: Direct sentiment (no summarization)
    title = video.get('title', '')
    if title:
        try:
            title_sentiment = sentiment_analyzer(title[:512])[0]
            video['title_sentiment'] = title_sentiment
        except Exception as e:
            if debug:
                print(f"    Title sentiment failed: {e}")
            video['title_sentiment'] = None
    else:
        video['title_sentiment'] = None
    
    # Transcript: Summarize → Sentiment
    transcript_text = video.get('transcript_text', '')
    
    if debug:
        print(f"   Transcript length: {len(transcript_text)} chars, {len(transcript_text.split())} words")
    
    # Check if transcript exists and is long enough
    if not transcript_text or len(transcript_text.strip()) < 200:
        if debug:
            print(f"    Transcript too short or missing")
        video['transcript_summary'] = None
        video['transcript_sentiment'] = None
        return video
    
    try:
        # Clean and truncate transcript
        transcript_text = transcript_text.strip()
        words = transcript_text.split()
        
        if debug:
            print(f"   Word count: {len(words)}")
        
        # BART works best with 100-1024 tokens
        if len(words) < 100:
            if debug:
                print(f"    Too few words: {len(words)}")
            video['transcript_summary'] = None
            video['transcript_sentiment'] = None
            return video
        
        if len(words) > 1000:
            transcript_text = ' '.join(words[:1000])
            if debug:
                print(f"    Truncated to 1000 words")
        
        if debug:
            print(f"   Generating summary...")
            print(f"   First 200 chars: {transcript_text[:200]}")
        
        # Generate summary with better parameters

        summary = summarize_long_text(transcript_text)
        
        if debug:
            print(f"   Summary: {summary}")
        
        # Sentiment of summary
        transcript_sentiment = sentiment_analyzer(summary[:512])[0]
        
        if debug:
            print(f"   Sentiment: {transcript_sentiment}")
        
        video['transcript_summary'] = summary
        video['transcript_sentiment'] = transcript_sentiment
        
    except Exception as e:
        if debug:
            print(f"   Error: {type(e).__name__}: {str(e)}")
        video['transcript_summary'] = None
        try:
            video['transcript_sentiment'] = sentiment_analyzer(transcript_text[:512])[0]
        except Exception:
            video['transcript_sentiment'] = None
    
    return video

# Test on first video with debug output
def get_all_sentiments(
    videos,
    debug_first=True,
    cache_path=SENTIMENT_CACHE_PATH,
    max_cache_size=MAX_SENTIMENT_CACHE_SIZE,
    save_cache=True,
):
    if debug_first and videos:
        print("\nTesting first video with debug output:\n")
        test_video = videos[0].copy()
        print(f"Title: {test_video.get('title')}")
        analyze_video_sentiment(test_video, debug=True)
        print("\n" + "="*80 + "\n")

    latest_ids = [v.get("video_id") for v in videos if v.get("video_id")]
    if cache_path:
        cache, cache_format = load_sentiment_cache(cache_path)
    else:
        cache, cache_format = {}, "map"
    cache = prune_sentiment_cache(cache, latest_ids, max_cache_size)

    print("\nAnalyzing all videos...")
    for video in tqdm(videos, desc="Processing videos", unit="video"):
        vid = video.get("video_id")
        if cache and vid in cache:
            cached = cache[vid] or {}
            video["title_sentiment"] = cached.get("title_sentiment")
            video["transcript_summary"] = cached.get("transcript_summary")
            video["transcript_sentiment"] = cached.get("transcript_sentiment")
            continue

        if video.get('transcript_text'):
            analyze_video_sentiment(video, debug=False)
        else:
            video['title_sentiment'] = None
            video['transcript_summary'] = None
            video['transcript_sentiment'] = None
        if cache_path:
            update_sentiment_cache(cache, video)

    print("Analysis complete!\n")
    if cache_path and save_cache:
        cache = prune_sentiment_cache(cache, latest_ids, max_cache_size)
        videos_to_save = videos
        if cache_format == "videos" and max_cache_size:
            keep_ids = set(latest_ids[:max_cache_size])
            videos_to_save = [v for v in videos if v.get("video_id") in keep_ids]
        save_sentiment_cache(
            cache_path,
            cache,
            videos=videos_to_save,
            cache_format=cache_format,
        )
    return videos
