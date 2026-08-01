from youtube_transcript_api import YouTubeTranscriptApi
from pytubefix import YouTube
from urllib.parse import urlparse, parse_qs

api = YouTubeTranscriptApi()

# Get video id from URL
def extract_video_id(url):
    parsed = urlparse(url)

    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")

    return parse_qs(parsed.query)["v"][0]

# Get transcript
def get_transcript(url):
    video_id = extract_video_id(url)

    try:
        # Get video transcript
        transcript = api.fetch(video_id, languages = ["en"])

        # Get transcript data
        transcript_data = []

        for snippet in transcript:
            transcript_data.append({
                "text": snippet.text,
                "start": snippet.start,
                "duration": snippet.duration
            })
        
        return transcript_data

    except Exception as e:
        raise RuntimeError(f"Failed to fetch transcript: {e}")

# Get video metadata
def get_metadata(url):
    yt = YouTube(url)

    metadata = {
        "source_type": "youtube",
        "video_title": yt.title,
        "author": yt.author,
        "url": url,
        "duration": yt.length,
        "upload_date": yt.publish_date,
    }

    return metadata

# Main function executing all calls
def load_youtube(url):
    return {
        "metadata": get_metadata(url),
        "transcript": get_transcript(url)
    }