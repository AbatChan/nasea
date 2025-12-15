# NASEA Web

Web interface for NASEA - works on any browser, any device.

## Quick Start (Local)

```bash
# From the nasea root directory
cd web
pip install chainlit
chainlit run app.py
```

Open http://localhost:8000

## Deploy to Hugging Face Spaces (Free)

1. Create a new Space at https://huggingface.co/spaces
2. Choose "Docker" as the SDK
3. Upload all files from this repo
4. Add secrets in Settings:
   - `VENICE_API_KEY` (or `OPENAI_API_KEY`)

## Deploy to Railway (Free tier)

1. Connect your GitHub repo at https://railway.app
2. Add environment variables:
   - `VENICE_API_KEY` (or `OPENAI_API_KEY`)
3. Deploy automatically

## Deploy to Render (Free)

1. Create new Web Service at https://render.com
2. Connect GitHub repo
3. Set:
   - Build Command: `pip install -r web/requirements.txt`
   - Start Command: `chainlit run web/app.py --host 0.0.0.0 --port $PORT`
4. Add environment variables

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VENICE_API_KEY` | Yes* | Venice AI API key |
| `OPENAI_API_KEY` | Yes* | OpenAI API key (alternative) |
| `TAVILY_API_KEY` | No | For better web search |

*At least one API key required

## Features

- Real-time streaming responses
- Tool execution with visual feedback
- Project creation with file display
- Web search integration
- Dark theme by default
