# Deployment

## Request path

```text
Browser -> FastAPI -> Agent Core -> model / tools / USDA -> JSON response
```

The browser submits same-origin `/chat` requests. FastAPI validates bounded
input and thread IDs, maps internal failures to sanitized HTTP errors, and
delegates all nutrition behavior to Agent Core.

## Local run

```bash
pip install -r requirements.txt
uvicorn web_app:app --host 127.0.0.1 --port 8000
```

`GET /healthz` returns readiness metadata without calling the model or USDA.
Ready services return HTTP 200; degraded services return HTTP 503. `POST /chat`
returns the answer, thread ID, runtime mode, architecture, memory mode, trace ID,
and safe tool-name evidence.

## Hosted backend

Set configuration in the hosting platform, never in source control:

- `CALORIECHEF_ARCHITECTURE=single`
- `CALORIECHEF_MODEL_BACKEND=hosted`
- `CALORIECHEF_HOSTED_MODEL`
- `OPENAI_API_KEY`
- `USDA_API_KEY`
- `CALORIECHEF_ENABLE_LONG_TERM_MEMORY=false`
- `CALORIECHEF_REQUEST_TIMEOUT_SECONDS`

Hosted mode never falls back to localhost Ollama. Long-term memory remains
limited because this package does not provide hosted embeddings or durable
vector storage. SQLite thread state is instance-local and can disappear after
a restart or redeploy.

## Render settings

| Setting | Value |
|---|---|
| Root directory | Repository root |
| Build command | `pip install -r requirements-deploy.txt` |
| Start command | `uvicorn web_app:app --host 0.0.0.0 --port $PORT` |
| Health-check path | `/healthz` |

## Public verification

After a real deployment, perform the three automated HTTP requests:

```bash
python scripts/verify_deployment.py --base-url <URL>
```

Then open the same URL in a browser and ask:

> How many calories and how much protein are in chicken breast?

Confirm a visible answer that identifies USDA as the source. In the browser
Network panel, confirm `tools_called` includes `search_food` and
`get_food_nutrition`. Save the URL, health JSON, chat JSON, and screenshot.

Public deployment remains pending until a real URL passes every check.
