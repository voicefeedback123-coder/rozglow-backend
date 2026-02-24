import os
import re
import json
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="RozGlow API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class ExtractRequest(BaseModel):
    url: str

class RemedyOut(BaseModel):
    title: str
    skin_concern: str
    ingredients: list[str]
    steps: list[str]
    duration: str
    frequency: str
    suitable_for: str
    caution: str

class ExtractResponse(BaseModel):
    video_title: str
    channel_name: str
    remedies: list[RemedyOut]

def extract_video_id(url: str) -> str:
    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract video ID from URL")

def get_transcript(video_id: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Method 1: Try youtube-transcript-api first
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=['hi', 'en', 'hi-IN', 'en-IN'])
        full_text = " ".join([snippet.text for snippet in transcript])
        if len(full_text.strip()) > 50:
            if len(full_text) > 12000:
                full_text = full_text[:12000] + "..."
            return full_text
    except Exception:
        pass
    
    # Method 2: Try yt-dlp to extract subtitles
    try:
        # First get available subs
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "--write-auto-sub", "--sub-lang", "hi,en,hi-IN,en-IN",
             "--sub-format", "json3", "--write-sub", "-o", "/tmp/%(id)s", url],
            capture_output=True, text=True, timeout=30
        )
        
        # Try to find and read the subtitle file
        import glob
        sub_files = glob.glob(f"/tmp/{video_id}*.json3") + glob.glob(f"/tmp/{video_id}*.vtt") + glob.glob(f"/tmp/{video_id}*.srv*")
        
        if sub_files:
            with open(sub_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse json3 format
            if sub_files[0].endswith('.json3'):
                data = json.loads(content)
                events = data.get('events', [])
                texts = []
                for event in events:
                    segs = event.get('segs', [])
                    for seg in segs:
                        text = seg.get('utf8', '').strip()
                        if text and text != '\n':
                            texts.append(text)
                full_text = " ".join(texts)
            else:
                # VTT/SRV format - strip tags and timestamps
                lines = content.split('\n')
                texts = []
                for line in lines:
                    line = line.strip()
                    if not line or '-->' in line or line.isdigit() or line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
                        continue
                    clean = re.sub(r'<[^>]+>', '', line)
                    if clean.strip():
                        texts.append(clean.strip())
                full_text = " ".join(texts)
            
            # Cleanup
            for f in sub_files:
                os.remove(f)
            
            if len(full_text.strip()) > 50:
                if len(full_text) > 12000:
                    full_text = full_text[:12000] + "..."
                return full_text
    except Exception:
        pass
    
    # Method 3: Use yt-dlp to get auto-generated subtitles differently
    try:
        result = subprocess.run(
            ["yt-dlp", "--skip-download", "--write-auto-sub", "--sub-lang", "hi,en",
             "--convert-subs", "vtt", "-o", "/tmp/sub_%(id)s", url],
            capture_output=True, text=True, timeout=30
        )
        
        import glob
        sub_files = glob.glob(f"/tmp/sub_{video_id}*")
        
        if sub_files:
            with open(sub_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
            lines = content.split('\n')
            texts = []
            for line in lines:
                line = line.strip()
                if not line or '-->' in line or line.isdigit() or line.startswith('WEBVTT') or line.startswith('Kind:') or line.startswith('Language:'):
                    continue
                clean = re.sub(r'<[^>]+>', '', line)
                if clean.strip():
                    texts.append(clean.strip())
            full_text = " ".join(texts)
            
            for f in sub_files:
                os.remove(f)
            
            if len(full_text.strip()) > 50:
                if len(full_text) > 12000:
                    full_text = full_text[:12000] + "..."
                return full_text
    except Exception:
        pass
    
    raise HTTPException(status_code=400, detail="Could not fetch transcript. Make sure the video has subtitles/captions enabled.")

SYSTEM_PROMPT = """You are a skincare remedy extraction AI for the RozGlow app.
Your job is to analyze YouTube video transcripts (often in Hindi/Hinglish/English) and extract structured home remedy information.

IMPORTANT RULES:
1. Only extract ACTUAL remedies mentioned in the video — do NOT invent remedies
2. If the transcript doesn't contain any skincare remedies, return an empty array
3. Translate Hindi/Hinglish content to English for the output
4. Keep ingredient names simple and commonly understood
5. Be specific with steps — users should be able to follow them
6. Include safety cautions where relevant (allergies, patch testing, etc.)

Return ONLY valid JSON in this exact format:
{
  "video_title": "Best guess at the video title from content",
  "channel_name": "Best guess or Unknown",
  "remedies": [
    {
      "title": "Short descriptive name for the remedy",
      "skin_concern": "Primary concern (e.g., Acne, Dark Spots, Glowing Skin, Tan Removal, Anti-Aging)",
      "ingredients": ["ingredient 1", "ingredient 2"],
      "steps": ["Step 1 description", "Step 2 description"],
      "duration": "How long to apply (e.g., 15-20 minutes)",
      "frequency": "How often to use (e.g., 2-3 times per week)",
      "suitable_for": "Skin types this works for",
      "caution": "Any warnings or patch test advice (empty string if none)"
    }
  ]
}"""

def extract_remedies_from_transcript(transcript: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all skincare home remedies from this video transcript:\n\n{transcript}"},
        ],
        temperature=0.3,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return json.loads(content)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "rozglow-api"}

@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    try:
        video_id = extract_video_id(req.url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    transcript = get_transcript(video_id)

    if len(transcript.strip()) < 50:
        raise HTTPException(status_code=400, detail="Video transcript is too short")

    try:
        result = extract_remedies_from_transcript(transcript)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid response. Please try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI extraction failed: {str(e)}")

    return ExtractResponse(
        video_title=result.get("video_title", ""),
        channel_name=result.get("channel_name", ""),
        remedies=[RemedyOut(**r) for r in result.get("remedies", [])],
    )
