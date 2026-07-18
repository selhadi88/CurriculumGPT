# CurriculumGPT

Curriculum-to-industry alignment platform. Given a list of courses/skills (an
individual's coursework, or a full institutional curriculum), it compares them
against real job postings using a transformer embedding model and reports:

- an overall alignment score, industry coverage, and curriculum relevance
- a skill-gap breakdown (which in-demand skills the curriculum doesn't cover)
- best-matching jobs, with matched vs. missing skills per job
- relevant certifications to close the identified gaps
- exportable PDF / Excel reports

The scoring core also implements a seven-component alignment model
(semantic, topic coverage, recency, cognitive depth, skill-graph, trend,
structure) described in the accompanying paper — see `ml/models/curriculum_gpt.py`.

## Quick start

```bash
docker compose up --build
```

This starts Postgres, the FastAPI backend (`:7000`), and the React frontend
(`:5173`). On first boot the backend downloads the pretrained
`BAAI/bge-large-en-v1.5` embedding model (~1.3GB) from Hugging Face — this
needs internet access and takes a few minutes. The database seeds itself
automatically from the sample data under `data/mappings/` and
`data/processed/` (a hand-authored demo dataset — see below).

Open `http://localhost:5173` once it's up.

No `.env` file is required to get started — `docker-compose.yml` has working
defaults for everything. Copy `.env.example` to `.env` if you want to add
real API keys (O*NET, USAJobs, Gemini/Groq) for the data-collection scripts.

## Project structure

```
backend/    FastAPI app: REST API, DB models, alignment service
frontend/   React + Vite + Tailwind UI
ml/         Models (BGE bi-encoder, 7-component CurriculumGPT), training, evaluation
data/       Skill taxonomy, certifications, sample O*NET-style occupations,
            and seed courses/jobs (small, hand-authored — see caveat below)
scripts/    CLI entry points for scraping, preprocessing, training, evaluation
```

## About the seed data

The `data/mappings/*.json` and `data/processed/**/*.csv` files in this repo
are **hand-authored sample data**, not scraped from live sources — enough to
demonstrate the full pipeline end-to-end (55 skills, 52 sample courses, 52
sample jobs across IT, engineering, finance, marketing, operations, HR, and
design). They are not a substitute for real, current labor-market data.

`scripts/scrape.py`, `scripts/preprocess.py`, and `scripts/build_dataset.py`
are designed to pull real data from O*NET, USAJobs, and other sources, but the
`data/scrapers/` and `data/preprocessing/` packages they depend on are not
included in this repository — that data-collection layer would need to be
built or restored before those scripts can run.

## Model modes

- `MODEL_TYPE=bge` (default): zero-shot cosine similarity on the pretrained
  BGE backbone. Works immediately, no training required.
- `MODEL_TYPE=curriculum_gpt`: the seven-component model. Requires a trained
  checkpoint (`MODEL_CHECKPOINT=...`) to be meaningful — with random,
  untrained weights it will not produce useful scores. No checkpoint ships
  with this repo.

## Known limitations

- The skill-gap/heatmap/radar views use a literal keyword-matching detector
  (`_detect_skills_keyword`) that is separate from the main semantic
  alignment score — it only catches phrases present in the taxonomy's
  keyword lists, so unusual phrasing can under-report curriculum coverage
  even when the semantic score correctly recognizes the match.
- No automated test suite exists yet (`pyproject.toml` points `pytest` at
  `backend/tests`, which doesn't exist).
- Alembic is wired up but has no migration history; tables are created
  directly from the SQLAlchemy models at startup.
