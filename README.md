# A place to think out loud

A supportive-listening chatbot with a two-tier crisis-detection layer in front of the model.

> **Not clinically validated.** This is a technical starting point. Do not put it in
> front of real users without a licensed mental-health professional reviewing
> `SYSTEM_PROMPT` and `crisis_reply()`. Depending on where you operate, software that
> resembles diagnosis or treatment may be a regulated medical device.

## Architecture

```
user message
   │
   ├─► TIER 1  regex prefilter        (instant, free, works if the API is down)
   ├─► TIER 2  Flash-Lite classifier  (catches indirect phrasing)
   └─► max(tier1, tier2)
          ├── acute   → fixed, reviewable crisis script. Model never runs.
          ├── concern → model + CONCERN_OVERLAY, exercises suppressed
          └── none    → model, normal system prompt
```

Three decisions worth keeping:

- **The acute path bypasses the LLM.** At the highest risk level you want auditable,
  non-drifting text. A generated response is one nobody can sign off on.
- **The classifier fails closed.** API error returns `concern`, never `none`.
- **Two tiers.** Regex misses "I've written letters to everyone." Models miss things
  when the API is down.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# paste your key into that file
streamlit run streamlit_app.py
```

## Deploy free (Streamlit Community Cloud)

1. Push this folder to a **public GitHub repo**. Confirm `.streamlit/secrets.toml`
   is NOT in it — `.gitignore` covers it, but check.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **Create app** → **Deploy a public app from GitHub**.
4. Repo = yours, Branch = `main`, Main file path = `streamlit_app.py`.
5. **Advanced settings** → Secrets → paste:
   ```toml
   GEMINI_API_KEY = "AIza..."
   ```
6. **Deploy.** First build takes a few minutes. You get a permanent
   `*.streamlit.app` URL.

To update: push to GitHub, the app redeploys automatically.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Everything, in numbered sections |
| `requirements.txt` | Dependencies |
| `.streamlit/secrets.toml` | Local key (gitignored, never committed) |

Sections 3 (prompts) and 4 (safety) are where nearly all your tuning happens.

## Known limitations

- **Flag log is ephemeral.** Community Cloud wipes the filesystem on restart.
  For real review, write to a database or Google Sheet instead.
- **No auth.** Anyone with the URL can use it, spending your quota.
- **Free Gemini tier trains on inputs.** Fine for your own test messages. Not
  acceptable for real users' disclosures — you need a paid tier with a no-training
  guarantee before launch, and the consent screen must say where data goes.
- **Rate limit is per session**, so it's trivially bypassed by opening a new tab.
  Real protection needs server-side limiting by IP or account.

## Before real users

- [ ] Clinician review of system prompt + crisis script
- [ ] Every helpline number dialled and verified; recheck quarterly
- [ ] Paid API tier with no-training guarantee
- [ ] Real storage with encryption at rest and a working delete
- [ ] Server-side rate limiting and abuse protection
- [ ] A human actually staffing the flagged-conversation queue
- [ ] Written incident plan for when the system misses a crisis
- [ ] Red-teamed by people who aren't you
