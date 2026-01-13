from googleapiclient.discovery import build
# import io
from dotenv import load_dotenv
from googleapiclient.discovery import build
import os
import json
load_dotenv()

YOUTUBE_API_KEY = os.getenv("API_KEY")
CHANNEL_ID = "UCrp_UI8XtuYfpiqluWLD7Lw"  # CNBC channel
MAX_VIDEOS = 100

def save_to_json(videos, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)

def fetch_youtube_videos_with_api(channel_id = CHANNEL_ID, api_key = YOUTUBE_API_KEY, max_results=100):
    """Fetch YouTube videos using Data API (no transcripts needed)"""
    youtube = build('youtube', 'v3', developerKey=api_key)
    videos = []
    next_page_token = None
    uploads_playlist_id = None
    
    print(f"Fetching videos from channel {channel_id}...")
    
    while len(videos) < max_results:
        try:
            # First, get the uploads playlist ID for the channel
            if uploads_playlist_id is None:  # Only need to do this once
                channel_response = youtube.channels().list(
                    part='contentDetails',
                    id=channel_id
                ).execute()
                
                if not channel_response.get('items'):
                    print(f"❌ Channel {channel_id} not found")
                    break
                
                uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Get videos from uploads playlist
            if uploads_playlist_id:
                request = youtube.playlistItems().list(
                    part='snippet,contentDetails',
                    playlistId=uploads_playlist_id,
                    maxResults=min(max_results, max_results - len(videos)),
                    pageToken=next_page_token
                )
            else:
                # Fallback: search for videos from channel
                request = youtube.search().list(
                    part='snippet',
                    channelId=channel_id,
                    type='video',
                    maxResults=min(max_results, max_results - len(videos)),
                    pageToken=next_page_token,
                    order='date'
                )
            
            response = request.execute()
            
            # Get video IDs
            video_ids = []
            for item in response['items']:
                if 'contentDetails' in item:
                    video_ids.append(item['contentDetails']['videoId'])
                elif 'id' in item and 'videoId' in item['id']:
                    video_ids.append(item['id']['videoId'])
            
            # Get detailed video information
            if video_ids:
                video_details = youtube.videos().list(
                    part='snippet,statistics',
                    id=','.join(video_ids)
                ).execute()
                
                for item in video_details['items']:
                    snippet = item['snippet']
                    videos.append({
                        'title': snippet.get('title', ''),
                        'video_id': item['id'],
                        'url': f"https://www.youtube.com/watch?v={item['id']}",
                        'published': snippet.get('publishedAt', ''),
                        'published_date': snippet.get('publishedAt', ''),
                        'author': snippet.get('channelTitle', ''),
                        'summary': snippet.get('description', ''),  # Full description
                        'transcript_text': None,  # No transcript (IP banned)
                        'view_count': item['statistics'].get('viewCount', 0),
                        'like_count': item['statistics'].get('likeCount', 0),
                    })
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
                
            print(f"  Fetched {len(videos)} videos so far...")
            
        except Exception as e:
            print(f"❌ Error fetching videos: {e}")
            break
    
    print(f"✅ Total videos fetched: {len(videos)}")
    save_to_json(videos, "100vids.json")
    return videos
