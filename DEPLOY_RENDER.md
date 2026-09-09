# Deploying CurriculumGPT to Render.com

This guide deploys the full stack from the GitHub repo
<https://github.com/selhadi88/CurriculumGPT> using the `render.yaml` Blueprint
at the repo root.

Resources created:

| Resource | Render type | Plan in `render.yaml` |
|---|---|---|
| `curriculumgpt-db` | PostgreSQL 16 | **free** |
| `curriculumgpt-backend` | Web Service (Docker) | **standard** ($25/mo) — see note |
| `curriculumgpt-frontend` | Static Site | **free** |

---

## ⚠️ Read this first: the backend is not free-tier friendly

The backend loads a **PyTorch BGE embedding model** into memory. Render's
**free and Starter web services cap RAM at 512 MB**, and the default model
(`BAAI/bge-large-en-v1.5`, ~1.3 GB on disk) needs **~2 GB RAM** to load and
run. On 512 MB the service is killed (OOM) during startup and never becomes
healthy.

Your options, cheapest first:

| Choice | Backend plan | `BGE_MODEL_NAME` | Reality |
|---|---|---|---|
| A | `starter` ($7/mo, 512 MB) | `BAAI/bge-small-en-v1.5` | Borderline — torch + small model + app is ~500–700 MB peak. May still OOM. Worth a try. |
| B | `standard` ($25/mo, 2 GB) | `BAAI/bge-large-en-v1.5` | Reliable. This is the `render.yaml` default. |
| C | `standard` ($25/mo, 2 GB) | `BAAI/bge-small-en-v1.5` | Reliable and cheaper on cold-start bandwidth; slightly lower scoring quality. |

The **database and frontend are genuinely free.** Render's free PostgreSQL is
also **deleted after 30 days** unless you upgrade it — fine for a demo, not for
anything you need to keep.

To use option A, edit `render.yaml` before deploying:

```yaml
  - type: web
    name: curriculumgpt-backend
    plan: starter          # was: standard
    ...
    envVars:
      - key: BGE_MODEL_NAME
        value: BAAI/bge-small-en-v1.5   # was: BAAI/bge-large-en-v1.5
```

---

## Step 1 — Push the repo (already done if Claude ran the commit)

`render.yaml` and the deployment fixes must be on GitHub `main`:

```bash
git add render.yaml DEPLOY_RENDER.md .env.example backend/ frontend/ ml/
git commit -m "Add Render deployment config"
git push origin main
```

## Step 2 — Create the Blueprint on Render

1. Sign in at <https://dashboard.render.com> (create an account / accept
   Render's terms yourself — Claude cannot do this).
2. **New ▸ Blueprint**.
3. Connect GitHub and pick **`selhadi88/CurriculumGPT`**, branch `main`.
   Authorize Render's GitHub app for that repo when prompted.
4. Render reads `render.yaml` and shows the 3 resources. Click **Apply**.
5. It will prompt for the `sync: false` env vars (see Step 3). You can leave
   them blank for now and click through.

## Step 3 — Environment variables

Most are set automatically by `render.yaml`. Set these manually **only if you
want the data-collection / LLM-labeling scripts to work** — the live web app
does **not** need any of them:

| Variable | Service | Where to get it | Needed for |
|---|---|---|---|
| `ONET_API_KEY` | backend | <https://services.onetcenter.org/> | `scripts/scrape.py` |
| `USAJOBS_API_KEY` | backend | <https://developer.usajobs.gov/> | `scripts/scrape.py` |
| `USAJOBS_EMAIL` | backend | your email | `scripts/scrape.py` |
| `GROQ_API_KEY` | backend | <https://console.groq.com/keys> | `scripts/build_dataset.py --llm` |
| `GROQ_API_KEY_2` | backend | (optional 2nd Groq key) | same, daily-quota fallback |
| `GEMINI_API_KEY` | backend | <https://aistudio.google.com/apikey> | LLM-labeling fallback |

Set automatically — do not touch:

| Variable | Service | Value |
|---|---|---|
| `DATABASE_URL` | backend | from `curriculumgpt-db` |
| `SECRET_KEY` | backend | auto-generated |
| `MODEL_TYPE` | backend | `bge` |
| `BGE_MODEL_NAME` | backend | `BAAI/bge-large-en-v1.5` |
| `MODEL_CACHE_DIR` | backend | `/model_cache` (persistent disk) |
| `DEVICE` | backend | `cpu` |
| `SEED_LIMIT` | backend | `2000` |
| `CORS_ORIGIN_REGEX` | backend | `https://.*\.onrender\.com` |
| `VITE_API_URL` | frontend | backend hostname, via `fromService` |

## Step 4 — First deploy

- **Backend** builds the Docker image (torch CPU download — 8–15 min the first
  time). On boot it downloads the BGE model to the `/model_cache` disk (once,
  then cached), runs `alembic upgrade head`, and seeds Postgres from the sample
  CSVs. Watch **Logs** for `Application startup complete`.
- **Frontend** runs `pnpm install && vite build` and publishes `frontend/dist`.
  It bakes in `VITE_API_URL` pointing at the backend.
- Health check: `https://curriculumgpt-backend-XXXX.onrender.com/api/v1/health`
  should return `{"status":"ok",...}`.

If the frontend built **before** the backend existed, its `VITE_API_URL` may be
empty. After both are up, open the frontend service ▸ **Manual Deploy ▸ Clear
build cache & deploy** so it picks up the backend URL.

## Step 5 — Verify

Open `https://curriculumgpt-frontend-XXXX.onrender.com`:

- Dashboard shows domain stats (proves frontend → backend → DB works).
- **Analyzer**: paste `Data Structures, Machine Learning, Databases, Statistics`,
  run an individual analysis, confirm a score and skill-gap table come back.
- Job Explorer lists sample jobs.

## Notes / gotchas

- **CORS**: `CORS_ORIGIN_REGEX=https://.*\.onrender\.com` lets any Render
  subdomain call the API, so you don't have to hardcode the frontend URL. To
  lock it down, replace it with an exact `CORS_ORIGINS_RAW=https://your-frontend.onrender.com`.
- **Cold starts**: free/starter services sleep after 15 min idle; the next
  request takes ~1 min (plus model reload if the disk is not used).
- **Custom domain**: add it on the *frontend* service, then also add it to
  `CORS_ORIGINS_RAW` on the backend.
- **`postgres://` vs `postgresql+psycopg2://`**: handled in code
  (`backend/app/config.py`, `backend/alembic/env.py`) — no action needed.
- **Local dev is unchanged**: `docker compose up --build -d` still works; the
  backend `CMD` falls back to port 7000 when `$PORT` is unset.
