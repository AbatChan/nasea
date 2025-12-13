# NASEA

NASEA is a terminal CLI that generates/edits projects from natural language prompts.

## Install (recommended)

Requirements:
- Python 3.11+
- At least one LLM API key

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Edit .env and add your API key(s)
```

## Run

```bash
nasea generate "Create a Spotify homepage clone"
```

Output goes to `./output` by default (configurable via `.env`).

## Workflow tips

- If you see `⚠ Task may be incomplete`, type `/continue` to resume with prior tool context.
- For web UI work, treat `index.html` as the source of truth for CSS/JS hooks (keeps class/id selectors aligned).

## Optional

- `node` (optional): improves JS syntax checking (`node --check`)
- Playwright (optional): enables `open_browser` previews

## Client-friendly setup

See `CLIENT_GUIDE.md`.

For issues or questions, contact: [Alex]

---

**Built with:** Python, LangChain, Kimi K2/GPT-4, pytest, typer

**Version:** 0.1.0-alpha (MVP)

**Last Updated:** November 2025
