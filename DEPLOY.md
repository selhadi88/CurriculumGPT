# Deploying CurriculumGPT

Two hosts, both free:

| Part | Host | What runs there | URL |
|---|---|---|---|
| 1 | Hugging Face **Docker Space** | FastAPI + PyTorch + `BAAI/bge-large-en-v1.5` | `https://selhadi-curriculumgpt-backend.hf.space` |
| 2 | Render **Static Site** | React/Vite UI | `https://curriculumgpt-frontend-XXXX.onrender.com` |
| — | Neon (or Supabase) | PostgreSQL 16 (HF Spaces has no managed DB) | internal |

Deploy **Part 1 first** — Part 2's `VITE_API_URL` points at it.

Config already in the repo:

```
.hf/Dockerfile                          HF Space image (bakes the model, non-root, port 7860)
.hf/README.md                           HF Space card (sdk: docker, app_port: 7860)
.github/workflows/deploy-hf-space.yml   GitHub Action: push main -> HF Space
render.yaml                             Render Blueprint for the static frontend
frontend/package.json                   packageManager: pnpm@9.15.4  (fixes ERR_PNPM_IGNORED_BUILDS)
```

---

## Part 0 — Database (Neon, ~3 min, free, no card)

1. Go to <https://neon.tech> → sign in with GitHub → **Create project**
   (name `curriculumgpt`, Postgres 16, region near you).
2. On the project dashboard, **Connection string** → copy the **pooled**
   connection string. It looks like:
   `postgresql://curriculumgpt_owner:XXXX@ep-xxx-pooler.us-east-2.aws.neon.tech/curriculumgpt?sslmode=require`
3. Keep it for Part 1. (Supabase → *Project Settings ▸ Database ▸ Connection string ▸ URI* works too.)

The app normalizes `postgresql://` → `postgresql+psycopg2://` itself and keeps
`?sslmode=require`, so paste the string as-is.

---

## Part 1 — Backend on Hugging Face Spaces

### 1.1 Create a Hugging Face access token

1. <https://huggingface.co/settings/tokens> → **Create new token**.
2. Type **Write**. Name it `curriculumgpt-deploy`. Create, copy the `hf_...` value.

### 1.2 Add it to GitHub

1. GitHub repo → **Settings ▸ Secrets and variables ▸ Actions ▸ New repository secret**.
2. Name `HF_TOKEN`, value = the `hf_...` token. **Add secret**.

### 1.3 Create the Space

1. <https://huggingface.co/new-space>.
2. **Owner** `selhadi` · **Space name** `curriculumgpt-backend`
   (this exact name → the URL `selhadi-curriculumgpt-backend.hf.space`).
3. **License** your choice · **SDK: Docker** → **Blank** template.
4. **Space hardware**: *CPU basic · 16 GB · free*.
5. Leave it **Public** (a private Space's `.hf.space` URL requires a token, which
   the browser frontend can't send). **Create Space.**

### 1.4 Set the Space variables & secrets

Space → **Settings ▸ Variables and secrets**:

**Secrets** (click *New secret*):

| Name | Value |
|---|---|
| `DATABASE_URL` | the Neon pooled connection string from Part 0 |
| `SECRET_KEY` | any long random string (e.g. `openssl rand -hex 32`) |

**Variables** (click *New variable*):

| Name | Value |
|---|---|
| `CORS_ORIGIN_REGEX` | `https://.*\.onrender\.com` |
| `SEED_LIMIT` | `2000` |

Optional secrets — only if you also want to run the scraping / LLM-labeling
scripts inside the Space; the web app does **not** need them:
`ONET_API_KEY`, `USAJOBS_API_KEY`, `USAJOBS_EMAIL`, `GROQ_API_KEY`,
`GROQ_API_KEY_2`, `GEMINI_API_KEY`.

Everything else (`MODEL_TYPE`, `BGE_MODEL_NAME`, `DEVICE`, `MODEL_CACHE_DIR`,
`HF_HOME`, `PORT`) is baked into `.hf/Dockerfile` — don't set it.

### 1.5 Trigger the deploy

The GitHub Action pushes the backend to the Space. Run it once now:

- GitHub repo → **Actions ▸ "Deploy backend to Hugging Face Space" ▸ Run workflow ▸ main**.
- (It also runs automatically on every future push to `main` that touches
  `backend/`, `ml/`, `data/`, or `.hf/`.)

Then watch **the Space's "Building" logs** on huggingface.co:

- First build ≈ 10–20 min (installs CPU torch, bakes the 1.3 GB model).
- When it flips to **"Running"**, check:
  `https://selhadi-curriculumgpt-backend.hf.space/api/v1/health`
  → `{"status":"ok","model_type":"bge","version":"0.1.0"}`
- API docs: `https://selhadi-curriculumgpt-backend.hf.space/docs`

First `/health` may lag a few seconds while the DB seeds (~4 000 rows, once).
The model loads lazily on the first `/api/v1/analyze` call (~15–30 s), then stays warm.

---

## Part 2 — Frontend on Render (free static site)

### 2.1 Create the static site

1. <https://dashboard.render.com> → sign in (create the account / accept
   Render's terms yourself).
2. **New ▸ Blueprint**. Connect GitHub, pick **`selhadi88/CurriculumGPT`**,
   branch `main`. Authorize Render's GitHub app for the repo.
3. Render reads `render.yaml` and shows **curriculumgpt-frontend** (static).
   Click **Apply**.

`render.yaml` already sets:

- **Build**: `cd frontend && corepack enable && pnpm install --no-frozen-lockfile && pnpm run build`
- **Publish dir**: `frontend/dist`
- **Env var** `VITE_API_URL = https://selhadi-curriculumgpt-backend.hf.space`
- SPA rewrite `/* → /index.html`

### 2.2 Prefer to click it in by hand instead of the Blueprint?

**New ▸ Static Site** → repo `selhadi88/CurriculumGPT` → then set:

| Field | Value |
|---|---|
| Branch | `main` |
| Root Directory | *(leave blank)* |
| Build Command | `cd frontend && corepack enable && pnpm install --no-frozen-lockfile && pnpm run build` |
| Publish Directory | `frontend/dist` |

Environment ▸ **Add Environment Variable**:

| Key | Value |
|---|---|
| `VITE_API_URL` | `https://selhadi-curriculumgpt-backend.hf.space` |

Redirects/Rewrites ▸ **Add Rule**: Source `/*`, Destination `/index.html`,
Action **Rewrite**.

Then **Create Static Site**.

### 2.3 Verify

Build takes ~2–3 min. Open `https://curriculumgpt-frontend-XXXX.onrender.com`:

- **Dashboard** shows job-domain stats → frontend → HF backend → Neon all wired.
- **Analyzer**: paste
  `Data Structures, Machine Learning, Databases, Statistics`,
  run an *individual* analysis → a score + skill-gap table come back.
- **Job Explorer** lists sample postings.

If the dashboard is empty, open the browser dev-tools **Network** tab: a failing
call to `…hf.space/api/v1/...` means the Space isn't "Running" yet, or CORS —
confirm `CORS_ORIGIN_REGEX=https://.*\.onrender\.com` is set on the Space and
the Space was restarted after you set it (Settings ▸ *Factory reboot* or push).

---

## Environment variables — quick reference

### Hugging Face Space (Settings ▸ Variables and secrets)

| Name | Kind | Value | Needed by |
|---|---|---|---|
| `DATABASE_URL` | secret | Neon pooled connection string | API (required) |
| `SECRET_KEY` | secret | long random string | API (required) |
| `CORS_ORIGIN_REGEX` | variable | `https://.*\.onrender\.com` | API (required for the browser) |
| `SEED_LIMIT` | variable | `2000` | DB seeding |
| `ONET_API_KEY` | secret | from services.onetcenter.org | scripts only |
| `USAJOBS_API_KEY` | secret | from developer.usajobs.gov | scripts only |
| `USAJOBS_EMAIL` | secret | your email | scripts only |
| `GROQ_API_KEY` / `GROQ_API_KEY_2` | secret | from console.groq.com | scripts only |
| `GEMINI_API_KEY` | secret | from aistudio.google.com | scripts only |

### GitHub repo (Settings ▸ Secrets and variables ▸ Actions)

| Name | Value |
|---|---|
| `HF_TOKEN` | Hugging Face **write** token |

### Render static site (Environment)

| Name | Value |
|---|---|
| `VITE_API_URL` | `https://selhadi-curriculumgpt-backend.hf.space` |

---

## Notes / gotchas

- **`ERR_PNPM_IGNORED_BUILDS`**: fixed by `"packageManager": "pnpm@9.15.4"` in
  `frontend/package.json` (corepack then uses that exact pnpm, which honors
  `pnpm.onlyBuiltDependencies: ["esbuild"]`). pnpm 10+ ignores that field, so
  the pin matters — don't bump it without also moving the allow-list into a
  `pnpm-workspace.yaml`.
- **HF free Spaces sleep** after 48 h idle and cold-start in ~30–60 s (the model
  is baked in, so no re-download). The Neon DB is unaffected.
- **Neon free** auto-suspends after 5 min idle and resumes on the next query
  (~1 s). Fine for a demo.
- **Custom domain** on the frontend: add it in Render, then also add
  `https://your-domain` to a `CORS_ORIGINS_RAW` variable on the Space (comma-
  separated; it's in addition to the regex).
- **Local dev is unchanged**: `docker compose up --build -d` still runs all
  three services locally on 5173 / 7000 / 5433.
- **Redeploying the backend**: push to `main` (or Actions ▸ Run workflow). The
  Space rebuilds automatically.
