# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "python-dotenv",
#   "fastapi",
#   "uvicorn[standard]",
#   "playwright",
#   "beautifulsoup4",
#   "pandas",
#   "pdfplumber",
#   "openpyxl",
#   "matplotlib"
# ]
# ///

"""
receive_requests.py

LLM-powered endpoint for the LLM Analysis Quiz project (MAX POWER / Option C).

High-level behavior
-------------------
- POST /receive_requests with JSON:
    { "email": "...", "secret": "...", "url": "https://..." }

- Validates:
    * 400 on invalid JSON
    * 403 on bad secret
    * 200 + {"status": "accepted"} on success (work done in background)

- Background worker:
    * Uses Playwright (Chromium headless) to render quiz page (JS, atob, etc.)
    * Extracts:
        - Quiz question text (Q###...)
        - <pre> JSON template, if present
        - Submit URL (any link containing "/submit")
        - File links (pdf/csv/xlsx/xls/json/txt/zip/images/audio)
    * Downloads files (httpx), saves to temp dir
    * For each file:
        - CSV/XLSX: pandas preview (table snippet)
        - PDF: pdfplumber text snippet
        - JSON/TXT: raw text snippet
        - IMAGES (png/jpg/jpeg/webp/gif): kept as image; sent to LLM via vision
        - AUDIO (mp3/wav/m4a/flac/ogg): transcribed with OpenAI whisper-1
    * Computes numeric fallback answer (sum of "value"-like columns) using Python.
    * Asks OpenAI LLM (gpt-5-omni-mini by default) to:
        - Understand instructions
        - Use provided snippets (tables, text, transcripts, images) to compute answer
        - Optionally design a chart spec
    * If LLM returns a chart spec:
        - Renders chart with matplotlib
        - Encodes PNG as data:image/png;base64,... and uses that as the "answer".
    * Builds final JSON payload:
        - Starts with <pre> JSON (if valid) as a template
        - Overrides: email, secret, url, answer
    * Submits payload to detected submit URL (httpx POST)
    * If response returns another URL (url / next_url), follows chain
      up to MAX_STEPS or MAX_TOTAL_SECONDS.

Important:
- NO generated code is executed.
- LLM is used for interpretation, reasoning, cleaning, analysis, vision & chart design.
"""

import os
import re
import json
import time
import base64
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

import httpx
import pdfplumber
import pandas as pd

# For chart rendering
import io
import matplotlib.pyplot as plt

# -------------------------------------------------
# Environment / config
# -------------------------------------------------
load_dotenv()

app = FastAPI()

# Secret used to validate incoming requests
SECRET_KEY = os.getenv("SECRET_KEY")  # e.g. SECRET_KEY=mysecret

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")  # medium-cost default

HTTP_TIMEOUT = 60.0
MAX_STEPS = 10              # max number of chained quiz URLs
MAX_TOTAL_SECONDS = 3 * 60  # 3-minute limit from problem statement

# Hard safety: don't call LLM if no key
USE_LLM = bool(OPENAI_API_KEY)

# Image/audio extensions
IMAGE_EXTS = {"png", "jpg", "jpeg", "webp", "gif"}
AUDIO_EXTS = {"mp3", "wav", "m4a", "flac", "ogg"}


# -------------------------------------------------
# Utility helpers
# -------------------------------------------------
def safe_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "download"
    return name


def ensure_absolute(url: str, base: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin(base, url)


async def http_get_bytes(
    url: str, client: httpx.AsyncClient, timeout: float = HTTP_TIMEOUT
) -> tuple[bytes, str]:
    resp = await client.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content, resp.headers.get("content-type", "")


# -------------------------------------------------
# File download & numeric parsing (fallback)
# -------------------------------------------------
async def download_file_to_temp(url: str, quiz_base_url: str) -> tuple[str, str]:
    """
    Download a file (supports relative URLs) and save it to a temp folder.
    Returns (local_path, extension_without_dot).
    """
    abs_url = ensure_absolute(url, quiz_base_url)
    print(f"Downloading: {abs_url}")

    async with httpx.AsyncClient() as client:
        content, content_type = await http_get_bytes(abs_url, client)

    tmpdir = tempfile.mkdtemp(prefix="llm_quiz_")
    filename = safe_filename_from_url(abs_url)
    local_path = os.path.join(tmpdir, filename)

    with open(local_path, "wb") as f:
        f.write(content)

    ext = Path(filename).suffix.lower().lstrip(".")
    if not ext and content_type:
        if "pdf" in content_type:
            ext = "pdf"
        elif "csv" in content_type:
            ext = "csv"
        elif "excel" in content_type or "spreadsheet" in content_type:
            ext = "xlsx"
        elif "json" in content_type:
            ext = "json"
        else:
            ext = content_type.split("/")[-1]

    print(f"Saved: {local_path} (ext: {ext})")
    return local_path, ext


def extract_numbers_from_string(s: str) -> list[float]:
    """
    Extract numeric tokens from text; handles commas, spaces, negatives, and (123) as -123.
    """
    nums: list[float] = []
    pattern = r"-?\(?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\)?|-?\d+\.\d+|-?\d+"
    for m in re.finditer(pattern, s):
        token = m.group(0)
        token = token.replace(",", "").replace(" ", "")
        if token.startswith("(") and token.endswith(")"):
            token = "-" + token[1:-1]
        try:
            nums.append(float(token))
        except Exception:
            continue
    return nums


def parse_pdf_for_values(path: str) -> dict:
    """
    Try to extract a numeric 'value' column from tables in a PDF.
    Fallback: all numbers in text.
    Returns: {"total": float | None, "details": [str, ...]}
    """
    details: list[str] = []
    total: float | None = None

    try:
        with pdfplumber.open(path) as pdf:
            found_values: list[float] = []

            # 1) Try tables
            for i, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if not tables:
                    continue

                for table in tables:
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                    except Exception:
                        df = pd.DataFrame(table)

                    lowered = [str(c).strip().lower() for c in df.columns]
                    value_cols = [idx for idx, name in enumerate(lowered) if "value" in name]
                    if value_cols:
                        for idx in value_cols:
                            ser = pd.to_numeric(
                                df.iloc[:, idx]
                                .astype(str)
                                .str.replace(",", "", regex=False)
                                .str.replace(" ", "", regex=False),
                                errors="coerce",
                            )
                            vals = ser.dropna().tolist()
                            if vals:
                                found_values.extend(vals)
                                details.append(f"pdf_page{i}_col{idx}_values:{len(vals)}")

            if found_values:
                total = float(sum(found_values))
                details.append(f"pdf_table_value_sum:{total}")
                return {"total": total, "details": details}

            # 2) Fallback: text numeric scan
            text_nums: list[float] = []
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                nums = extract_numbers_from_string(text)
                if nums:
                    text_nums.extend(nums)
                    details.append(f"pdf_page{i}_text_nums:{len(nums)}")

            if text_nums:
                total = float(sum(text_nums))
                details.append(f"pdf_text_sum:{total}")
                return {"total": total, "details": details}

    except Exception as e:
        details.append(f"pdf_error:{e}")

    return {"total": total, "details": details}


def parse_csv_xlsx_for_values(path: str) -> dict:
    """
    Parse CSV or XLSX file, prefer a column named like 'value'.
    If missing, pick the numeric column with highest sum.
    Returns: {"total": float | None, "details": [str, ...]}
    """
    details: list[str] = []
    total: float | None = None

    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path, dtype=str)
        else:
            df = pd.read_excel(path, dtype=str, engine="openpyxl")

        cols = {c: str(c).strip().lower() for c in df.columns}
        value_cols = [orig for orig, low in cols.items() if "value" in low]

        # Case 1: we have explicit 'value' columns
        if value_cols:
            sums: list[tuple[str, float]] = []
            for c in value_cols:
                ser = pd.to_numeric(
                    df[c]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace(" ", "", regex=False),
                    errors="coerce",
                )
                s = float(ser.sum(skipna=True) or 0.0)
                details.append(f"col:{c}_sum:{s}")
                sums.append((c, s))

            total = float(sum(s for _, s in sums))
            details.append(f"value_cols_total:{total}")
            return {"total": total, "details": details}

        # Case 2: guess best numeric column
        numeric_sums: list[tuple[str, float]] = []
        for c in df.columns:
            ser = pd.to_numeric(
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False),
                errors="coerce",
            )
            if ser.notna().any():
                numeric_sums.append((c, float(ser.sum(skipna=True) or 0.0)))

        if numeric_sums:
            numeric_sums.sort(key=lambda x: x[1], reverse=True)
            best_col, best_sum = numeric_sums[0]
            total = float(best_sum)
            details.append(f"best_numeric_col:{best_col}_sum:{best_sum}")
            return {"total": total, "details": details}

        details.append("no_numeric_columns_found")
        return {"total": None, "details": details}

    except Exception as e:
        details.append(f"csv_xlsx_error:{e}")
        return {"total": None, "details": details}


def parse_json_generic(path: str) -> dict:
    """
    Very generic JSON handling: if we find a top-level list of objects with a 'value'
    field, we sum that. Otherwise we sum all numeric values we can find.
    """
    details: list[str] = []
    total: float | None = None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        nums: list[float] = []

        def walk(obj):
            if isinstance(obj, dict):
                for _, v in obj.items():
                    if isinstance(v, (int, float)):
                        nums.append(float(v))
                    else:
                        walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        # Prefer top-level list with 'value'
        if isinstance(data, list) and data and isinstance(data[0], dict) and "value" in data[0]:
            for row in data:
                try:
                    v = row.get("value")
                    if v is not None:
                        nums.append(float(v))
                except Exception:
                    continue
            details.append("json_value_field_sum")
        else:
            walk(data)
            details.append("json_all_numeric_sum")

        if nums:
            total = float(sum(nums))
            details.append(f"json_total:{total}")
        else:
            details.append("json_no_numeric_data")

    except Exception as e:
        details.append(f"json_error:{e}")

    return {"total": total, "details": details}


# -------------------------------------------------
# Audio transcription (OpenAI Whisper)
# -------------------------------------------------
async def transcribe_audio_file(path: str) -> str | None:
    """
    Use OpenAI whisper-1 to transcribe an audio file.
    """
    if not USE_LLM:
        print("Audio transcription skipped: no OPENAI_API_KEY set.")
        return None

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(path, "rb") as f:
                files = {"file": (os.path.basename(path), f, "audio/mpeg")}
                data = {"model": "whisper-1"}
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
                resp = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files=files,
                    data=data,
                )
        if resp.status_code != 200:
            print("⚠️ Whisper non-200 response:", resp.status_code, resp.text[:300])
            return None
        body = resp.json()
        text = body.get("text")
        print("✔ Audio transcript length:", len(text) if text else 0)
        return text
    except Exception as e:
        print("⚠️ Audio transcription failed:", e)
        return None


# -------------------------------------------------
# Build snippets for LLM (so it can reason on data)
# -------------------------------------------------
def build_file_snippet_for_llm(path: str, ext: str, max_chars: int = 8000, max_rows: int = 100) -> dict:
    """
    Build a small, LLM-friendly summary of the file contents.

    We intentionally limit size to keep token usage reasonable.
    """
    info: dict = {"type": ext, "name": os.path.basename(path)}

    try:
        if ext in ("csv", "tsv"):
            df = pd.read_csv(path)
            head = df.head(max_rows)
            info["table_preview"] = head.to_dict(orient="records")
            info["columns"] = list(df.columns)
            info["row_count_estimate"] = int(len(df))
        elif ext in ("xls", "xlsx"):
            df = pd.read_excel(path, engine="openpyxl")
            head = df.head(max_rows)
            info["table_preview"] = head.to_dict(orient="records")
            info["columns"] = list(df.columns)
            info["row_count_estimate"] = int(len(df))
        elif ext == "pdf":
            text = ""
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages[:5]:
                    text += (page.extract_text() or "") + "\n"
                    if len(text) >= max_chars:
                        break
            info["text_preview"] = text[:max_chars]
        elif ext == "json":
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read(max_chars)
            info["text_preview"] = raw
        elif ext in IMAGE_EXTS:
            # We won't put base64 here; images are passed via vision in ask_llm_for_answer.
            info["image_file"] = True
        elif ext in AUDIO_EXTS:
            # Transcript (if available) is populated in parsing; we'll just mark type.
            info["audio_file"] = True
        else:
            with open(path, "rb") as f:
                raw = f.read(max_chars)
            info["text_preview"] = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        info["error"] = f"snippet_error:{e}"

    return info


# -------------------------------------------------
# Chart rendering
# -------------------------------------------------
def render_chart_to_data_uri(chart_spec: dict) -> str | None:
    """
    Render a simple chart from a spec into a data:image/png;base64 URI.

    Expected chart_spec structure:
    {
      "kind": "bar" | "line" | "pie",
      "title": "string",
      "x": [...],
      "y": [...],
      "xlabel": "string",
      "ylabel": "string"
    }
    """
    try:
        kind = (chart_spec.get("kind") or "bar").lower()
        x = chart_spec.get("x") or []
        y = chart_spec.get("y") or []
        title = chart_spec.get("title") or "Chart"
        xlabel = chart_spec.get("xlabel") or ""
        ylabel = chart_spec.get("ylabel") or ""

        fig, ax = plt.subplots()

        if kind == "line":
            ax.plot(x, y)
        elif kind == "pie":
            if not x:
                x = [f"v{i}" for i in range(len(y))]
            ax.pie(y, labels=x)
        else:
            ax.bar(x, y)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print("⚠️ Chart rendering failed:", e)
        return None


# -------------------------------------------------
# OpenAI LLM call (with optional vision images)
# -------------------------------------------------
async def ask_llm_for_answer(
    question: str,
    page_context: str,
    pre_json_template: dict | None,
    parsed_results: list[dict],
    numeric_fallback: float | None,
    feedback: dict | None = None,
) -> dict | None:
    """
    Ask OpenAI to compute the answer based on:
    - quiz question
    - page context
    - file snippets
    - (optional) pre_json_template
    - (optional) numeric_fallback from our own parsing
    - (optional) feedback from previous failed submission

    Returns a dict like:
    {
      "answer": ...,
      "answer_type": "number" | "string" | "boolean" | "object" | "chart_spec" | "file_base64",
      "reasoning": "short text",
      "chart": { ... } | null
    }
    or None if anything fails.
    """
    if not USE_LLM:
        print("LLM disabled (no OPENAI_API_KEY set). Skipping LLM step.")
        return None

    # Build file snippets for LLM
    llm_files: list[dict] = []
    for r in parsed_results:
        path = r.get("path")
        ext = r.get("ext") or r.get("type")
        if not path or not ext:
            continue
        snippet = build_file_snippet_for_llm(path, ext)
        snippet["parser_summary"] = r.get("parsing")
        llm_files.append(snippet)

    # Keep page context limited to avoid massive prompts
    page_context_short = page_context[:6000]

    payload_for_llm = {
        "question": question,
        "page_context": page_context_short,
        "pre_json_template": pre_json_template,
        "numeric_fallback": numeric_fallback,
        "files": llm_files,
        "feedback": feedback,
    }

    system_prompt = (
        "You are an expert data, ML, and visualization assistant helping to solve a quiz.\n"
        "You receive:\n"
        "- A question and HTML page context (human-readable instructions).\n"
        "- Structured snippets of data files (tables, text, JSON, audio transcripts, images).\n"
        "- Sometimes a JSON template that shows how the answer should be posted.\n"
        "Carefully read the question and use ONLY the provided context and data.\n"
        "You MUST return a STRICT JSON object with this schema:\n"
        "{\n"
        "  \"answer\": <ANY VALID JSON VALUE>,\n"
        "  \"answer_type\": \"number\" | \"string\" | \"boolean\" | \"object\" | \"chart_spec\" | \"file_base64\",\n"
        "  \"reasoning\": \"short explanation of how you computed the answer\",\n"
        "  \"chart\": null OR {\n"
        "      \"kind\": \"bar\" | \"line\" | \"pie\",\n"
        "      \"title\": \"string\",\n"
        "      \"x\": [ ... ],\n"
        "      \"y\": [ ... ],\n"
        "      \"xlabel\": \"string\",\n"
        "      \"ylabel\": \"string\"\n"
        "  }\n"
        "}\n"
        "If no chart is needed, set chart = null and choose answer_type != \"chart_spec\".\n"
        "If the quiz explicitly asks for a chart or visualization as the answer, set:\n"
        "  answer_type = \"chart_spec\" and fill chart with a small, clean spec.\n"
        "If only a number/string/boolean/JSON is needed, ignore chart.\n"
        "Do NOT include markdown, comments, or any text outside the JSON object.\n"
    )

    user_prompt_text = (
        "Here is the quiz context and file data. Compute the correct answer.\n\n"
        "If the question expects a numeric result, return a number in the 'answer' field.\n"
        "If it expects text or JSON, return that.\n\n"
        "CONTEXT PAYLOAD:\n"
        + json.dumps(payload_for_llm, indent=2)
    )

    # Prepare image parts for vision (if any images were downloaded)
    image_parts = []
    max_images = 3
    for r in parsed_results:
        ext = (r.get("ext") or "").lower()
        if ext in IMAGE_EXTS and len(image_parts) < max_images:
            path = r.get("path")
            if not path:
                continue
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                image_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{ext if ext != 'jpg' else 'jpeg'};base64,{b64}"
                        },
                    }
                )
            except Exception as e:
                print("⚠️ Failed to load image for vision:", path, e)

    # User message content: text + optional images
    user_message_content: list[dict] = [{"type": "text", "text": user_prompt_text}]
    user_message_content.extend(image_parts)

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "temperature": 0.1,
                    "max_completion_tokens": 1500,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message_content},
                    ],
                },
            )
    except Exception as e:
        print("❌ LLM request failed:", e)
        return None

    if resp.status_code != 200:
        print("❌ LLM non-200 response:", resp.status_code, resp.text[:500])
        return None

    try:
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        # Extract JSON object (non-greedy)
        m = re.search(r"\{[\s\S]*\}", content)
        json_text = m.group(0) if m else content.strip()
        obj = json.loads(json_text)
        return obj
    except Exception as e:
        print("❌ Failed to parse LLM JSON:", e, "raw:", resp.text[:500])
        return None


# -------------------------------------------------
# Page parsing
# -------------------------------------------------
def extract_quiz_info(html: str, quiz_url: str) -> dict:
    """
    Given rendered HTML and quiz_url, extract:
      - quiz_question (if any)
      - submit_url
      - file_links (list)
      - pre_json_template (dict or None)
      - page_text (for fallback numeric scan)
    """
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)

    # Try question like "Q834. ..."
    q_match = re.search(r"Q\d+\.[^\n\r]+", html)
    quiz_question = q_match.group(0).strip() if q_match else "Unknown"

    # Collect URLs
    found_urls: set[str] = set()

    # All hrefs
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("javascript:"):
            continue
        found_urls.add(href)

    # Any explicit http(s) tokens inside the HTML
    found_urls.update(re.findall(r"https?://[^\s\"'<>]+", html))

    resolved_urls: list[str] = []
    for u in found_urls:
        try:
            resolved_urls.append(ensure_absolute(u, quiz_url))
        except Exception:
            continue

    # File links
    file_links: list[str] = []
    for u in resolved_urls:
        if re.search(r"\.(pdf|csv|xlsx|xls|json|txt|zip|png|jpg|jpeg|webp|gif|mp3|wav|m4a|flac|ogg)(?:[?#].*)?$", u, re.I):
            file_links.append(u)

    # Submit URL: any URL containing "/submit"
    submit_url: str | None = None
    for u in resolved_urls:
        # In weird cases the href may contain HTML; strip it
        plain = BeautifulSoup(u, "html.parser").get_text()
        if "/submit" in plain:
            submit_url = plain
            break

    # Pre JSON template
    pre_block = soup.find("pre")
    pre_json_template: dict | None = None
    pre_json_text: str | None = None
    if pre_block:
        pre_json_text = pre_block.get_text("\n", strip=True)
        try:
            pre_json_template = json.loads(pre_json_text)
        except Exception:
            pre_json_template = None

    return {
        "quiz_question": quiz_question,
        "submit_url": submit_url,
        "file_links": file_links,
        "pre_json_template": pre_json_template,
        "pre_json_text": pre_json_text,
        "page_text": page_text,
    }


# -------------------------------------------------
# Core chain runner
# -------------------------------------------------
async def process_request(data: dict):
    """
    Top-level background workflow. Follows a chain of quiz URLs
    until no new URL is returned, MAX_STEPS is reached, or time limit exceeded.
    """
    email = data.get("email")
    secret = data.get("secret")
    first_url = data.get("url")

    if not first_url:
        print("No 'url' provided in request; aborting.")
        return

    print(f"Processing request for: {email} starting at {first_url}")
    start_time = time.time()
    current_url = first_url
    step = 0
    overall_results: list[dict] = []

    async with async_playwright() as p:

        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-software-rasterizer",
                "--no-zygote",
                "--single-process"
            ]
        )

        page = await browser.new_page()


        while (
            current_url
            and step < MAX_STEPS
            and (time.time() - start_time) < MAX_TOTAL_SECONDS
        ):
            step += 1
            print(f"\n=== Step {step} — Visiting {current_url} ===")

            # Render page
            try:
                await page.goto(current_url, wait_until="networkidle", timeout=60000)
                full_html = await page.content()
            except Exception as e:
                print(f"⚠️ Playwright error while loading {current_url}:", e)
                full_html = ""

            # Try atob(`...`) decode
            decoded_html: str | None = None
            soup = BeautifulSoup(full_html, "html.parser")
            for script in soup.find_all("script"):
                code = script.string or ""
                if "atob(" in code:
                    m = re.search(r"atob\(`([^`]+)`\)", code, re.S)
                    if m:
                        b64 = m.group(1).replace("\n", "")
                        try:
                            decoded_html = base64.b64decode(b64).decode(
                                "utf-8", errors="replace"
                            )
                            print("✔ Decoded base64 inner HTML")
                            break
                        except Exception:
                            decoded_html = None
            if decoded_html is None:
                decoded_html = full_html

            quiz_info = extract_quiz_info(decoded_html, current_url)
            quiz_question = quiz_info["quiz_question"]
            submit_url = quiz_info["submit_url"]
            file_links = quiz_info["file_links"]
            pre_json_template = quiz_info["pre_json_template"]
            pre_json_text = quiz_info["pre_json_text"]
            page_text = quiz_info["page_text"]

            print("Question:", quiz_question)
            print("Submit URL:", submit_url)
            print("File links:", file_links)
            if pre_json_text:
                print("Pre JSON template:\n", pre_json_text)

            # ------------- Parse files (numeric + snippets) -------------
            parsed_results: list[dict] = []

            if not file_links:
                print("No file links detected on this page.")
            else:
                for file_url in file_links:
                    try:
                        local_path, ext = await download_file_to_temp(
                            file_url, current_url
                        )
                    except Exception as e:
                        print("❌ File download error:", e)
                        continue

                    parsing: dict | None = None
                    ext_lower = ext.lower()

                    if ext_lower == "pdf":
                        parsing = parse_pdf_for_values(local_path)
                    elif ext_lower in ("csv", "tsv", "txt"):
                        parsing = parse_csv_xlsx_for_values(local_path)
                    elif ext_lower in ("xls", "xlsx"):
                        parsing = parse_csv_xlsx_for_values(local_path)
                    elif ext_lower == "json":
                        parsing = parse_json_generic(local_path)
                    elif ext_lower in AUDIO_EXTS:
                        transcript = await transcribe_audio_file(local_path)
                        parsing = {
                            "total": None,
                            "details": ["audio_transcript"],
                            "transcript": transcript,
                        }
                    elif ext_lower in IMAGE_EXTS:
                        # Vision handled later; here just mark as image.
                        parsing = {
                            "total": None,
                            "details": ["image_file"],
                        }
                    else:
                        # generic text scan
                        try:
                            with open(local_path, "rb") as f:
                                raw = f.read(200000)
                            text = raw.decode("utf-8", errors="ignore")
                            nums = extract_numbers_from_string(text)
                            parsing = {
                                "total": float(sum(nums)) if nums else None,
                                "details": [f"generic_text_scan_nums:{len(nums)}"],
                            }
                        except Exception as e:
                            parsing = {
                                "total": None,
                                "details": [f"unknown_file_parse_error:{e}"],
                            }

                    parsed_results.append(
                        {"path": local_path, "ext": ext_lower, "parsing": parsing}
                    )

            # ------------- Numeric fallback answer -------------
            numeric_fallback: float | None = None
            parsing_details: list[str] = []

            for r in parsed_results:
                pinfo = r.get("parsing") or {}
                if pinfo.get("total") is not None:
                    numeric_fallback = float(pinfo["total"])
                    parsing_details.append(
                        f"{r['ext']}:{r['path']} -> {pinfo['total']}"
                    )
                    break

            if numeric_fallback is None:
                nums = extract_numbers_from_string(page_text)
                if nums:
                    numeric_fallback = float(sum(nums))
                    parsing_details.append(
                        f"page_text_fallback_count:{len(nums)} total:{numeric_fallback}"
                    )

            if numeric_fallback is None:
                print("ℹ No numeric fallback answer computed yet.")
            else:
                print("ℹ Numeric fallback answer:", numeric_fallback)

            # ------------- LLM answer (primary) -------------
            llm_result = await ask_llm_for_answer(
                question=quiz_question,
                page_context=page_text,
                pre_json_template=pre_json_template,
                parsed_results=parsed_results,
                numeric_fallback=numeric_fallback,
                feedback=None,
            )

            final_answer = None
            answer_source = "none"
            chart_spec = None

            if llm_result is not None:
                print("✔ LLM returned:", llm_result)
                answer_source = "llm"
                final_answer = llm_result.get("answer")
                answer_type = llm_result.get("answer_type", "auto")
                chart_spec = llm_result.get("chart")

                # If LLM says chart_spec, try to render chart to image and use that as answer
                if answer_type == "chart_spec" and chart_spec:
                    uri = render_chart_to_data_uri(chart_spec)
                    if uri:
                        final_answer = uri
                    else:
                        print("⚠️ Chart generation failed; using LLM 'answer' field directly.")
            else:
                print("⚠️ LLM did not return a usable JSON object.")

            # If LLM failed or didn't give an answer, fall back to numeric
            if final_answer is None and numeric_fallback is not None:
                final_answer = numeric_fallback
                answer_source = "numeric_fallback"

            if final_answer is None:
                print("❌ Could not compute any final answer (LLM + fallback failed).")
            else:
                print(f"✔ Final answer ({answer_source}):", final_answer)

            # ------------- Build final payload -------------
            if pre_json_template and isinstance(pre_json_template, dict):
                final_payload = dict(pre_json_template)  # shallow copy
            else:
                final_payload = {}

            final_payload["email"] = email
            final_payload["secret"] = secret
            final_payload["url"] = current_url
            final_payload["answer"] = final_answer

            print("Final payload to submit:", final_payload)

            # ------------- Submit answer -------------
            submit_result: dict | None = None
            if submit_url:
                try:
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                        resp = await client.post(submit_url, json=final_payload)
                        try:
                            resp.raise_for_status()
                            try:
                                submit_result = resp.json()
                            except Exception:
                                submit_result = {
                                    "status": "ok",
                                    "raw_text": resp.text,
                                }
                        except Exception as e:
                            submit_result = {
                                "status": "error",
                                "detail": str(e),
                                "raw_text": resp.text,
                            }
                except Exception as e:
                    submit_result = {"status": "error", "detail": str(e)}
            else:
                submit_result = {
                    "status": "skipped",
                    "reason": "no submit URL found on page",
                }

            print("Submit result:", submit_result)

            overall_results.append(
                {
                    "quiz_url": current_url,
                    "answer_source": answer_source,
                    "final_answer": final_answer,
                    "parsing_details": parsing_details,
                    "submit_result": submit_result,
                }
            )

            # ------------- Find next URL, if any -------------
            next_url: str | None = None

            if isinstance(submit_result, dict):
                # Official quiz spec: often returns `url` or `next_url`
                if submit_result.get("url"):
                    next_url = str(submit_result["url"])
                elif submit_result.get("next_url"):
                    next_url = str(submit_result["next_url"])

                # sometimes next payload is embedded in HTML inside raw_text
                if not next_url:
                    raw_text = submit_result.get("raw_text")
                    if raw_text and "<pre" in raw_text:
                        soup2 = BeautifulSoup(raw_text, "html.parser")
                        pre2 = soup2.find("pre")
                        if pre2:
                            pre_txt2 = pre2.get_text("\n", strip=True)
                            try:
                                j2 = json.loads(pre_txt2)
                                next_url = j2.get("url")
                            except Exception:
                                m = re.search(r"https?://[^\s\"'<>]+", pre_txt2)
                                if m:
                                    next_url = m.group(0)

            # If still nothing, see if this page's own <pre> JSON has a url
            if not next_url and pre_json_text:
                try:
                    j = json.loads(pre_json_text)
                    if j.get("url"):
                        next_url = j["url"]
                except Exception:
                    m = re.search(r"https?://[^\s\"'<>]+", pre_json_text)
                    if m:
                        next_url = m.group(0)

            if next_url:
                next_url = ensure_absolute(next_url, current_url)
                print("➡ Next URL:", next_url)
            else:
                print("No next URL found; stopping chain.")
                current_url = None
                break

            current_url = next_url

        await browser.close()

    print("\n=== Quiz chain finished ===")
    for r in overall_results:
        print(r)


# -------------------------------------------------
# FastAPI route
# -------------------------------------------------
@app.post("/receive_requests")
async def receive_requests(request: Request, background_tasks: BackgroundTasks):
    # 1) Validate JSON
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(content={"error": "invalid JSON"}, status_code=400)

    # 2) Validate secret
    if data.get("secret") != SECRET_KEY:
        return JSONResponse(content={"error": "Invalid secret"}, status_code=403)

    # 3) Launch async processing
    background_tasks.add_task(process_request, data)

    return JSONResponse(content={"status": "accepted", "data": data}, status_code=200)


# -------------------------------------------------
# Local server boot
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    print("Loaded SECRET_KEY:", SECRET_KEY)
    print("OPENAI_MODEL:", OPENAI_MODEL)
    print("OPENAI_API_KEY set?:", bool(OPENAI_API_KEY))
    uvicorn.run(app, host="0.0.0.0", port=8000)
