import os
import json
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
    transcript: str
    video_title: str = ""
    channel_name: str = ""


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


SYSTEM_PROMPT = (
    "You are a skincare remedy extraction AI for the RozGlow app. "
    "Analyze YouTube video transcripts (often Hindi/Hinglish/English) "
    "and extract structured home remedy information.\n\n"
    "RULES:\n"
    "1. Only extract ACTUAL remedies from the video - do NOT invent remedies\n"
    "2. If no skincare remedies found, return empty remedies array\n"
    "3. Translate Hindi/Hinglish to English\n"
    "4. Be specific with steps - users should be able to follow them\n"
    "5. Include safety cautions where relevant\n"
    "6. Guess the video title and channel from context if not provided\n\n"
    "Return ONLY valid JSON in this format:\n"
    '{"video_title":"...","channel_name":"...",'
    '"remedies":[{"title":"...","skin_concern":"...","ingredients":["..."],'
    '"steps":["..."],"duration":"...","frequency":"...",'
    '"suitable_for":"...","caution":"..."}]}'
)


def extract_remedies(transcript, video_title="", channel_name=""):
    user_msg = f"Extract all skincare home remedies from this transcript:\n\n"
    if video_title:
        user_msg += f"Video Title: {video_title}\n"
    if channel_name:
        user_msg += f"Channel: {channel_name}\n"
    user_msg += f"\nTranscript:\n{transcript}"

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
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
    transcript = req.transcript.strip()

    if len(transcript) < 50:
        raise HTTPException(
            status_code=400,
            detail="Transcript is too short. The video may not have captions."
        )

    # Truncate to 12000 chars to stay within token limits
    if len(transcript) > 12000:
        transcript = transcript[:12000] + "..."

    try:
        result = extract_remedies(transcript, req.video_title, req.channel_name)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid response. Please try again."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI extraction failed: {str(e)}"
        )

    return ExtractResponse(
        video_title=result.get("video_title", req.video_title),
        channel_name=result.get("channel_name", req.channel_name),
        remedies=[RemedyOut(**r) for r in result.get("remedies", [])],
    )
