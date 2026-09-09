---
title: CurriculumGPT Backend
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: FastAPI + BGE embedding API for curriculum-to-industry alignment
---

# CurriculumGPT Backend

FastAPI service that scores curricula against real job postings with a
`BAAI/bge-large-en-v1.5` embedding model. This Space hosts only the API; the
web UI is deployed separately (Render static site).

- API docs: `/docs`
- Health: `/api/v1/health`

Source and full project: <https://github.com/selhadi88/CurriculumGPT>

This Space repo is generated automatically from `main` by a GitHub Action —
do not edit it directly.

## Required configuration (Space → Settings → Variables and secrets)

| Name | Kind | Value |
|---|---|---|
| `DATABASE_URL` | secret | PostgreSQL connection string (e.g. from Neon) |
| `SECRET_KEY` | secret | any long random string |
| `CORS_ORIGIN_REGEX` | variable | `https://.*\.onrender\.com` |
| `SEED_LIMIT` | variable | `2000` |

Optional secrets (only for the data-collection / LLM-labeling scripts, not the
live API): `ONET_API_KEY`, `USAJOBS_API_KEY`, `USAJOBS_EMAIL`, `GROQ_API_KEY`,
`GROQ_API_KEY_2`, `GEMINI_API_KEY`.
