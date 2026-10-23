import os
import re
import json
import csv
import uuid
import threading
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify, send_file, abort

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from openai import OpenAI

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY",)

BASE_DIR = Path(__file__).resolve().parent
TRANSCRIPT_FOLDER = BASE_DIR / "youtube_transcripts"
JOBS_FOLDER = BASE_DIR / "jobs"
TRANSCRIPT_FOLDER.mkdir(parents=True, exist_ok=True)
JOBS_FOLDER.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

app = Flask(__name__)

# In-memory job registry (job_id -> status dict)
JOBS: dict[str, dict] = {}


# ============================================================
# YOUTUBE HELPERS (from reference 1)
# ============================================================
def extract_video_id(url: str) -> str:
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url if "://" in url else "https://" + url)
    host = parsed.netloc.lower().split(":")[0]

    if host in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.strip("/").split("/")[0]

    elif host in (
        "youtube.com", "www.youtube.com", "m.youtube.com",
        "music.youtube.com", "youtube-nocookie.com",
        "www.youtube-nocookie.com",
    ):
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            m = re.match(r"^/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})", parsed.path)
            video_id = m.group(1) if m else ""
    else:
        video_id = ""

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(f"Invalid YouTube URL: {url}")
    return video_id


def clean_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def fetch_transcript(video_id: str) -> tuple[str, str]:
    """Returns (full_text, language)."""
    api = YouTubeTranscriptApi()
    try:
        transcript = api.fetch(video_id, languages=["en", "hi"])
    except NoTranscriptFound:
        available = list(api.list(video_id))
        if not available:
            raise RuntimeError("No transcript available.")
        transcript = available[0].fetch()

    full_text = " ".join(
        clean_text(seg.text) for seg in transcript if clean_text(seg.text)
    )
    if not full_text:
        raise RuntimeError("Transcript was empty.")
    return full_text, transcript.language


# ============================================================
# DEEPSEEK HEADLINE EXTRACTION (from reference 2)
# ============================================================
SYSTEM_PROMPT = (
    "You are a professional news editor. You extract distinct news headlines "
    "from a news transcript. Return ONLY valid JSON — an array of strings. "
    "No markdown fences, no commentary."
)

USER_PROMPT_TEMPLATE = """Read the following news transcript and extract every distinct news story as a headline.

Rules:
- Each headline should be one concise sentence summarizing one story.
- Include names, places, numbers, and key facts where relevant.
- Preserve the order in which stories appear.
- Do NOT merge separate stories into one headline.
- Output ONLY a JSON array of strings, e.g.:
  ["Headline one...", "Headline two...", "Headline three..."]

TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"
"""


def extract_headlines(transcript: str) -> list[str]:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(transcript=transcript)},
        ],
        temperature=0.2,
        max_tokens=4000,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        headlines = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            raise ValueError(f"Model did not return valid JSON:\n{raw}")
        headlines = json.loads(m.group(0))

    if not isinstance(headlines, list) or not all(isinstance(h, str) for h in headlines):
        raise ValueError("Model output is not a list of strings.")

    return [h.strip() for h in headlines if h.strip()]


# ============================================================
# DEDUPLICATION  (unique headlines across all videos)
# ============================================================
def normalize_for_dedup(headline: str) -> str:
    """Lowercase, strip punctuation, collapse spaces — for near-dup detection."""
    s = headline.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def dedupe_headlines(headlines: list[str]) -> list[str]:
    """Remove exact & near-duplicate headlines, preserving order."""
    seen: set[str] = set()
    unique: list[str] = []
    for h in headlines:
        key = normalize_for_dedup(h)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(h)
    return unique


# ============================================================
# BACKGROUND WORKER
# ============================================================
def process_job(job_id: str, urls: list[str]):
    job = JOBS[job_id]
    all_headlines: list[str] = []
    per_video = []

    try:
        for idx, url in enumerate(urls, start=1):
            step = f"[{idx}/{len(urls)}] {url}"
            job["progress"] = f"Processing {step}"

            try:
                vid = extract_video_id(url)
                job["progress"] = f"[{idx}/{len(urls)}] Fetching transcript for {vid}..."
                text, lang = fetch_transcript(vid)

                # Optional: save raw transcript for audit
                (TRANSCRIPT_FOLDER / f"{vid}_transcript.txt").write_text(
                    text, encoding="utf-8"
                )

                job["progress"] = f"[{idx}/{len(urls)}] Extracting headlines from {vid}..."
                heads = extract_headlines(text)

                per_video.append({
                    "url": url,
                    "video_id": vid,
                    "language": lang,
                    "headlines": heads,
                })
                all_headlines.extend(heads)
                job["videos_done"] = idx

            except Exception as e:
                per_video.append({
                    "url": url,
                    "error": str(e),
                    "headlines": [],
                })
                job["errors"].append(f"{url} -> {e}")

        # Deduplicate across all videos
        unique = dedupe_headlines(all_headlines)

        # Write CSV
        # csv_path = JOBS_FOLDER / f"{job_id}.csv"
        # with csv_path.open("w", encoding="utf-8", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["id", "headline"])
        #     for i, h in enumerate(unique, start=1):
        #         writer.writerow([i, h])
                # Write CSV  →  col1 = headline, col2 = 1 (no header row)
        csv_path = JOBS_FOLDER / f"{job_id}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for h in unique:
                # 1 => america mornings (category = america)
                writer.writerow([h, 1])

        # Write JSON (audit / debugging)
        (JOBS_FOLDER / f"{job_id}.json").write_text(
            json.dumps(
                {
                    "created": datetime.utcnow().isoformat() + "Z",
                    "sources": per_video,
                    "unique_headlines": unique,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        job["status"] = "done"
        job["result"] = {
            "total_headlines": len(all_headlines),
            "unique_headlines": len(unique),
            "csv_filename": f"{job_id}.csv",
            "errors": job["errors"],
        }
        job["progress"] = "Complete."

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["progress"] = f"Failed: {e}"


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start_job():
    data = request.get_json(silent=True) or {}
    urls = [u.strip() for u in data.get("urls", []) if u and u.strip()]
    if not urls:
        return jsonify({"error": "Please provide at least one YouTube link."}), 400

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "status": "running",
        "progress": "Starting...",
        "videos_total": len(urls),
        "videos_done": 0,
        "errors": [],
        "result": None,
    }

    thread = threading.Thread(target=process_job, args=(job_id, urls), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(job)


@app.route("/api/download/<job_id>")
def download(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        abort(404, description="File not ready.")
    path = JOBS_FOLDER / f"{job_id}.csv"
    if not path.exists():
        abort(404)
    return send_file(
        path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"news_headlines_{job_id}.csv",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)