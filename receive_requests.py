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
receive_requests.py — Option B Agent Mode (Pure LLM + Tool Fetch)
The LLM is the core solver. It can request the backend to fetch URLs (JSON or text).
Retries up to 3 LLM attempts per quiz step. Returns only the LLM's "final_answer".
"""

import os
import re
import json
import time
import base64
from datetime import datetime
from urllib.parse import urljoin

import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

import httpx

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
load_dotenv()

app = FastAPI()

SECRET_KEY = os.getenv("SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
HTTP_TIMEOUT = 60
MAX_STEPS = 10
MAX_TOTAL_SECONDS = 180
MAX_LLM_ATTEMPTS = 3  # per quiz step

USE_LLM = bool(OPENAI_API_KEY)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def ensure_absolute(url, base):
    if not url:
        return url
    if url.startswith("http"):
        return url
    return urljoin(base, url)

def extract_numbers(s):
    if not s:
        return []
    pattern = r"-?\d+(?:\.\d+)?"
    return [float(x) for x in re.findall(pattern, s)]

async def http_fetch_raw(url, headers=None, timeout=HTTP_TIMEOUT):
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "")

async def http_fetch_text(url, headers=None, timeout=HTTP_TIMEOUT):
    content, ctype = await http_fetch_raw(url, headers=headers, timeout=timeout)
    try:
        return content.decode("utf-8", "ignore"), ctype
    except:
        return content, ctype

async def http_fetch_json(url, headers=None, timeout=HTTP_TIMEOUT):
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.json()

# ------------------------------------------------------------
# Page parsing (fast + robust)
# ------------------------------------------------------------
def extract_quiz_info(html, quiz_url):
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)

    # Quick extraction heuristics
    quiz_question = None
    tag = soup.select_one(".question, h1.question, h2.question, h3.question")
    if tag:
        quiz_question = tag.get_text(" ", strip=True)

    if not quiz_question:
        for h in soup.find_all(["h1", "h2", "h3"]):
            t = h.get_text(" ", strip=True)
            if not t:
                continue
            if "?" in t or t.lower().startswith(("what", "find", "calculate", "compute", "determine", "which", "how")):
                quiz_question = t
                break

    if not quiz_question:
        pre = soup.find("pre")
        if pre:
            raw = pre.get_text(" ", strip=True)
            try:
                j = json.loads(raw)
                for key in ("question", "prompt", "task", "q", "query"):
                    if key in j and isinstance(j[key], str):
                        quiz_question = j[key].strip()
                        break
            except:
                pass

    if not quiz_question:
        m = re.search(r"atob\(\s*`([^`]+)`\s*\)", html)
        if m:
            try:
                decoded = base64.b64decode(m.group(1)).decode("utf-8", "ignore")
                qm = re.search(r"Q\d+\.[^\n\r]{3,200}", decoded)
                if qm:
                    quiz_question = qm.group(0).strip()
                elif "?" in decoded:
                    quiz_question = decoded.split("?")[0] + "?"
            except:
                pass

    if not quiz_question:
        for s in re.split(r"[.?!]\s+", page_text):
            s = s.strip()
            if len(s) > 10 and ("?" in s or s.lower().startswith(("what", "find", "calculate", "compute", "determine"))):
                quiz_question = s
                break

    if not quiz_question:
        quiz_question = "Unknown"

    # submit url detection
    submit_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "submit" in href.lower():
            submit_url = ensure_absolute(href, quiz_url)
            break
        txt = a.get_text(" ", strip=True).lower()
        if "submit" in txt:
            submit_url = ensure_absolute(href, quiz_url)
            break

    if not submit_url:
        for u in re.findall(r"https?://[^\s\"'<]+", html):
            if "submit" in u.lower():
                submit_url = u
                break

    if not submit_url:
        m = re.search(r"fetch\(['\"]([^'\"]*submit[^'\"]*)['\"]\)", html, re.I)
        if m:
            submit_url = ensure_absolute(m.group(1), quiz_url)

    # file links
    urls = set()
    for a in soup.find_all("a", href=True):
        urls.add(a["href"])
    urls.update(re.findall(r"https?://[^\s\"'<]+", html))
    abs_urls = [ensure_absolute(u, quiz_url) for u in urls]

    file_links = [
        u for u in abs_urls
        if re.search(r"\.(pdf|csv|xlsx|xls|json|txt|png|jpg|jpeg|webp|gif)$", u, re.I)
    ]

    # pre json template
    pre_json = None
    pre_json_text = None
    pre = soup.find("pre")
    if pre:
        pre_json_text = pre.get_text("\n", strip=True)
        try:
            pre_json = json.loads(pre_json_text)
        except:
            pre_json = None

    return {
        "quiz_question": quiz_question,
        "submit_url": submit_url,
        "file_links": file_links,
        "pre_json_template": pre_json,
        "pre_json_text": pre_json_text,
        "page_text": page_text
    }

# ------------------------------------------------------------
# LLM Agent functions
# ------------------------------------------------------------
def build_agent_system_message():
    # instruct the LLM how to act and how to use the fetch tool
    return (
        "You are an expert data-processing agent. You will be given:\n"
        "- the quiz question (text)\n"
        "- the visible page text and any URLs found on the page\n\n"
        "You may request the system to fetch external resources using JSON objects ONLY with these forms:\n\n"
        "1) Fetch raw text/content:\n"
        '{ "action": "fetch", "url": "https://example.com/api" }\n\n'
        "2) Fetch and parse JSON (preferred for APIs returning JSON):\n"
        '{ "action": "fetch_json", "url": "https://example.com/api/data" }\n\n'
        "When you request a fetch, the system will perform it and then provide the response data back to you.\n\n"
        "When you have enough data to answer, RETURN EXACTLY one JSON object and NOTHING ELSE, with one of these forms:\n\n"
        "Success (final answer):\n"
        '{ \"final_answer\": <value>, \"explanation\": \"short explanation (optional)\" }\n\n'
        "Or, if you still need data to proceed, request fetches as above.\n\n"
        "Important rules:\n"
        " - Output must be valid JSON only. No surrounding text.\n"
        " - Use fetch_json for endpoints that return JSON. Use fetch for raw text or CSV.\n"
        " - If numerical precision is requested, return numbers (not strings).\n"
        " - Prefer programmatic exact results over vague language.\n"
        " - You have up to 3 attempts; if an attempt fails due to malformed JSON, retry more explicitly.\n"
    )

async def ask_llm_agent(conversation_messages):
    """
    conversation_messages: list of messages in openai chat format: [{"role":"system","content":...}, ...]
    Returns parsed JSON object from the assistant's content, or None on failure.
    """
    if not USE_LLM:
        return None

    async with httpx.AsyncClient(timeout=90) as client:
        payload = {
            "model": OPENAI_MODEL,
            "temperature": 0.0,
            "max_completion_tokens": 1500,
            "messages": conversation_messages
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        r = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        if r.status_code != 200:
            print("LLM error:", r.status_code, r.text[:400])
            return None
        j = r.json()
        content = j["choices"][0]["message"]["content"]
        # try to extract JSON object from content
        m = re.search(r"(\{[\s\S]*\})", content)
        if not m:
            # sometimes LLM returns raw JSON without braces detection - try entire content
            try:
                return json.loads(content)
            except Exception as e:
                print("LLM returned non-JSON content:", content[:400])
                return None
        try:
            return json.loads(m.group(1))
        except Exception as e:
            print("Failed to parse LLM JSON:", e, "raw:", m.group(1)[:800])
            return None

# ------------------------------------------------------------
# Background worker
# ------------------------------------------------------------
async def process_request(data):
    email = data.get("email")
    secret = data.get("secret")
    first_url = data.get("url")

    print("\n=== Processing:", email, first_url)

    start = time.time()
    current_url = first_url
    step = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--no-zygote",
                "--single-process",
                "--disable-dev-tools",
            ]
        )
        page = await browser.new_page()
        likely_selectors = [".question", "#result", "h2", "h1", "[data-question]", ".task"]

        system_msg = build_agent_system_message()

        while current_url and step < MAX_STEPS and time.time() - start < MAX_TOTAL_SECONDS:
            step += 1
            print(f"\n--- Step {step} | {current_url}")

            # Navigate and wait for content
            try:
                await page.goto(current_url, wait_until="domcontentloaded", timeout=60000)
            except Exception as e:
                print("Playwright goto error:", e)
                break

            # quick selector waits
            found_selector = False
            for sel in likely_selectors:
                try:
                    await page.wait_for_selector(sel, timeout=900)
                    found_selector = True
                    break
                except:
                    continue

            if not found_selector:
                await page.wait_for_timeout(500)
                try:
                    body_text = await page.evaluate("() => document.body ? document.body.innerText : ''")
                except Exception as e:
                    print("evaluate error:", e)
                    body_text = ""
                if body_text and len(body_text.strip()) > 20:
                    html = f"<html><body><pre>{body_text[:20000]}</pre></body></html>"
                else:
                    await page.wait_for_timeout(300)
                    try:
                        html = await page.content()
                    except Exception as e:
                        print("Playwright content error:", e)
                        html = ""
            else:
                await page.wait_for_timeout(200)
                try:
                    html = await page.content()
                except Exception as e:
                    print("Playwright content error:", e)
                    html = ""

            if not html or not html.strip():
                print("EMPTY HTML — stopping.")
                break

            info = extract_quiz_info(html, current_url)
            print("Question:", info["quiz_question"])
            print("Submit URL:", info["submit_url"])
            if info["quiz_question"] == "Unknown":
                print("DEBUG snippet:", (info["page_text"] or "")[:600].replace("\n", " "))

            # compute numeric fallback
            numeric_fallback = None
            nums = extract_numbers(info.get("page_text") or "")
            if nums:
                numeric_fallback = sum(nums)

            # Prepare initial conversation for LLM
            conversation = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps({
                    "question": info["quiz_question"],
                    "page_text": info.get("page_text"),
                    "file_links": info.get("file_links"),
                    "pre_json_template": info.get("pre_json_template"),
                    "instructions": "Use the fetch tool to retrieve external data as needed. Return final_answer JSON when ready."
                })}
            ]

            final_answer = None
            llm_attempt = 0

            # Agent loop: allow the LLM to ask for fetches and to return final_answer
            while llm_attempt < MAX_LLM_ATTEMPTS:
                llm_attempt += 1
                print(f"LLM attempt {llm_attempt} for step {step}...")
                res = await ask_llm_agent(conversation)
                if not res:
                    print("LLM produced no usable JSON response.")
                    # Add a hint and retry
                    conversation.append({"role": "assistant", "content": json.dumps({"error": "no_json_returned", "hint": "Please respond with JSON only. Use fetch or fetch_json if you need external data."})})
                    continue

                # If LLM asks to fetch
                if res.get("action") in ("fetch", "fetch_json") and res.get("url"):
                    fetch_url = ensure_absolute(res["url"], current_url)
                    print("LLM requested fetch:", res.get("action"), fetch_url)
                    try:
                        if res.get("action") == "fetch_json":
                            try:
                                fetched_json = await http_fetch_json(fetch_url)
                                fetched = {"url": fetch_url, "type": "json", "data": fetched_json}
                                conversation.append({"role": "assistant", "content": json.dumps({"fetched_url": fetch_url, "fetched_type": "json", "fetched_data": fetched_json})})
                                # Immediately continue loop; LLM will get next round with this data
                                continue
                            except Exception as e:
                                # try text fallback
                                txt, ctype = await http_fetch_text(fetch_url)
                                conversation.append({"role": "assistant", "content": json.dumps({"fetched_url": fetch_url, "fetched_type": "text", "fetched_data": txt[:100000], "error": str(e)})})
                                continue
                        else:
                            txt, ctype = await http_fetch_text(fetch_url)
                            # limit size to avoid huge messages
                            safe_txt = txt[:200000] if isinstance(txt, str) else txt
                            conversation.append({"role": "assistant", "content": json.dumps({"fetched_url": fetch_url, "fetched_type": "text", "fetched_data": safe_txt})})
                            continue
                    except Exception as e:
                        print("Fetch error:", e)
                        conversation.append({"role": "assistant", "content": json.dumps({"fetched_url": fetch_url, "error": str(e)})})
                        continue

                # If LLM provided final answer
                if "final_answer" in res:
                    final_answer = res.get("final_answer")
                    explanation = res.get("explanation")
                    print("LLM final_answer found:", final_answer)
                    # record explanation if present
                    if explanation:
                        print("LLM explanation:", (explanation[:400] + "...") if len(explanation) > 400 else explanation)
                    break

                # If LLM returned something else (unrecognized), provide stricter guidance and retry
                conversation.append({"role": "assistant", "content": json.dumps({"error": "unrecognized_response", "raw": res, "hint": "Return either {\"action\":\"fetch\",\"url\":\"...\"} or {\"final_answer\":...}."})})
                print("LLM returned unrecognized object; retrying...")

            # If LLM never returned final_answer, use numeric fallback or empty string
            if final_answer is None:
                if numeric_fallback is not None:
                    final_answer = numeric_fallback
                    print("Using numeric fallback:", final_answer)
                else:
                    final_answer = ""
                    print("No final answer from LLM; submitting empty string.")

            # Build payload
            payload = {}
            if info.get("pre_json_template") and isinstance(info["pre_json_template"], dict):
                payload = dict(info["pre_json_template"])
                if "answer" in payload:
                    payload["answer"] = final_answer
                else:
                    # try common keys
                    for candidate in ("answer", "result", "value", "response", "result_value"):
                        if candidate in payload:
                            payload[candidate] = final_answer
                            break
                    else:
                        payload["answer"] = final_answer
            else:
                payload = {
                    "email": email,
                    "secret": secret,
                    "url": current_url,
                    "answer": final_answer
                }

            payload.setdefault("email", email)
            payload.setdefault("secret", secret)
            payload.setdefault("url", current_url)

            # Submit
            submit_url = info.get("submit_url")
            next_url = None
            if submit_url:
                try:
                    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                        r = await client.post(submit_url, json=payload)
                        try:
                            jr = r.json()
                        except:
                            jr = {}
                    print("Submit response:", jr)
                    next_url = jr.get("url") or jr.get("next_url")
                except Exception as e:
                    print("Submit error:", e)
                    next_url = None
            else:
                print("No submit URL found on page.")
                next_url = None

            if next_url:
                current_url = ensure_absolute(next_url, current_url)
            else:
                print("No next URL. Stopping.")
                break

        await browser.close()

    print("\n=== Chain Finished ===")

# ------------------------------------------------------------
# FastAPI endpoint
# ------------------------------------------------------------
@app.post("/receive_requests")
async def receive_requests(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except:
        return JSONResponse({"error": "invalid JSON"}, 400)

    if data.get("secret") != SECRET_KEY:
        return JSONResponse({"error": "Invalid secret"}, 403)

    background_tasks.add_task(process_request, data)
    return {"status": "accepted", "data": data}

# ------------------------------------------------------------
# Local run
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("Loaded SECRET_KEY:", SECRET_KEY)
    print("OPENAI_MODEL:", OPENAI_MODEL)
    print("OpenAI Key Set:", bool(OPENAI_API_KEY))
    uvicorn.run(app, host="0.0.0.0", port=8000)
