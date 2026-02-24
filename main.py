import os
import re
import json
import glob
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


def extract_video_id(url):
    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise ValueError("Bad URL")


def cleanup_files(video_id):
    for f in glob.glob(f"/tmp/*{video_id}*"):
        try:
            os.remove(f)
        except Exception:
            pass


def parse_vtt(content):
    lines = content.split('\n')
    texts = []
    seen = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if '-->' in line:
            continue
        if line.isdigit():
            continue
        if line.startswith(('WEBVTT', 'Kind:', 'Language:', 'NOTE')):
            continue
        clean = re.sub(r'<[^>]+>', '', line).strip()
        if clean and clean not in seen:
            seen.add(clean)
            texts.append(clean)
    return " ".join(texts)


def truncate(text, limit=12000):
    text = text.strip()
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def get_transcript(video_id):
    clean_url = f"https://www.youtube.com/watch?v={video_id}"
    cleanup_files(video_id)

    # Method 1: youtube-transcript-api
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        ytt = YouTubeTranscriptApi()
        t = ytt.fetch(video_id, languages=['hi', 'en', 'hi-IN', 'en-IN'])
        txt = " ".join([s.text for s in t])
        if len(txt.strip()) > 50:
            return truncate(txt)
    except Exception:
        pass

    # Method 2: yt-dlp auto-generated subtitles
    try:
        prefix = f"/tmp/yta_{video_id}"
        subprocess.run(
            ["yt-dlp", "--skip-download", "--write-auto-sub",
             "--sub-lang", "hi,en", "--sub-format", "vtt",
             "--convert-subs", "vtt", "-o", prefix, clean_url],
            capture_output=True, text=True, timeout=45
        )
        for sf in glob.glob(f"{prefix}*.vtt"):
            with open(sf, 'r', encoding='utf-8') as fh:
                txt = parse_vtt(fh.read())
            cleanup_files(video_id)
            if len(txt.strip()) > 50:
                return truncate(txt)
    except Exception:
        pass

    # Method 3: yt-dlp manual subtitles
    try:
        prefix = f"/tmp/ytm_{video_id}"
        subprocess.run(
            ["yt-dlp", "--skip-download", "--write-sub",
             "--sub-lang", "hi,en", "--sub-format", "vtt",
             "--convert-subs", "vtt", "-o", prefix, clean_url],
            capture_output=True, text=True, timeout=45
        )
        for sf in glob.glob(f"{prefix}*.vtt"):
            with open(sf, 'r', encoding='utf-8') as fh:
                txt = parse_vtt(fh.read())
            cleanup_files(video_id)
            if len(txt.strip()) > 50:
                return truncate(txt)
    except Exception:
        pass

    cleanup_files(video_id)
    raise HTTPException(
        status_code=400,
        detail="Could not fetch transcript. Make sure the video has subtitles/captions enabled."
    )


SYSTEM_PROMPT = (
    "You are a skincare remedy extraction AI for the RozGlow app. "
    "Analyze YouTube video transcripts (often Hindi/Hinglish/English) "
    "and extract structured home remedy information.\n\n"
    "RULES:\n"
    "1. Only extract ACTUAL remedies from the video - do NOT invent remedies\n"
    "2. If no skincare remedies found, return empty array\n"
    "3. Translate Hindi/Hinglish to English\n"
    "4. Be specific with steps\n"
    "5. Include safety cautions\n\n"
    'Return ONLY valid JSON: {"video_title":"...","channel_name":"...",'
    '"remedies":[{"title":"...","skin_concern":"...","ingredients":["..."],'
    '"steps":["..."],"duration":"...","frequency":"...",'
    '"suitable_for":"...","caution":"..."}]}'
)


def extract_remedies(transcript):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all skincare home remedies:\n\n{transcript}"},
        ],
        temperature=0.3,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content)


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
        raise HTTPException(status_code=400, detail="Transcript too short")

    try:
        result = extract_remedies(transcript)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid response")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI failed: {str(e)}")

    return ExtractResponse(
        video_title=result.get("video_title", ""),
        channel_name=result.get("channel_name", ""),
        remedies=[RemedyOut(**r) for r in result.get("remedies", [])],
    )
