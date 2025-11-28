# LLM Quiz Solver API (FastAPI + Playwright + OpenAI GPT-5)

This Hugging Face Space exposes a `/receive_requests` endpoint that:
- Accepts quiz tasks via POST
- Scrapes the quiz webpage using Playwright
- Downloads PDF / CSV / XLSX / JSON / images / audio files
- Extracts tables and text
- Uses OpenAI GPT-5.1 (or other selected model)
- Optionally transcribes audio (Whisper-1)
- Optionally analyzes images (vision)
- Computes the quiz answer
- Submits the answer to the quiz’s submit URL
- Follows chained quiz URLs until completion

## Running locally
uvicorn receive_requests:app --reload --host 0.0.0.0 --port 8000

## Environment variables (set in Hugging Face → Settings → Variables & Secrets)
OPENAI_API_KEY  = Your OpenAI key
SECRET_KEY      = Any secret string used by incoming requests
OPENAI_MODEL    = gpt-5.1 (or any supported model)

## API Endpoint
POST to:
https://<your-space-name>.hf.space/receive_requests

## Example JSON Input
{
  "email": "abc@example.com",
  "secret": "mysecret",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}

The worker runs in background and automatically:
- Loads the webpage
- Scrapes & extracts data
- Processes files (PDF/CSV/XLSX/JSON/images/audio)
- Sends structured context to GPT-5
- Builds final answer
- Posts the answer to the quiz's submit URL
- Follows chained quiz URLs until completion
