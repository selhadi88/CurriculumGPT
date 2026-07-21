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

The `data/mappings/*.json` and `data/processed/{courses,jobs}/sample_*.csv`
files in this repo are **hand-authored sample data**, not scraped from live
sources — enough to demonstrate the full pipeline end-to-end (55 skills, 52
sample courses, 52 sample jobs across IT, engineering, finance, marketing,
operations, HR, and design) without needing any API keys.

## Real data sources (`data/scrapers/`)

Beyond the seed data above, `data/scrapers/` implements real collectors
against genuine public sources. Run via `scripts/scrape.py --source {onet,jobs,mit_ocw}`
(`jobs` covers both USAJobs and RemoteOK) or `--source all`. Output lands in
`data/raw/{courses,jobs}/*.jsonl`; `scripts/preprocess.py` (backed by
`data/preprocessing/pipeline.py`) then cleans, language-filters, deduplicates,
and flattens metadata into `data/processed/{courses,jobs}/*.csv`.

| Source | What it collects | Auth needed | License / terms |
|---|---|---|---|
| **O*NET Web Services** (`onet.py`) | Occupations + skills + technology skills, one query per taxonomy domain, deduped by O*NET-SOC code | `ONET_API_KEY` — free registration at [services.onetcenter.org](https://services.onetcenter.org/). Current accounts use the v2.0 API at `api-v2.onetcenter.org` with an `X-API-Key` header — the legacy `services.onetcenter.org/ws/` host uses old-style Basic Auth and is a different system entirely, confirmed the hard way while building this. | See [services.onetcenter.org/help/license_data](https://services.onetcenter.org/help/license_data) — O\*NET data is sponsored by the U.S. Department of Labor; check that page for the current attribution/usage terms before redistributing collected data. |
| **USAJobs** (`jobs.py`) | Federal job postings, keyword-swept per taxonomy domain | `USAJOBS_API_KEY` + `USAJOBS_EMAIL` — free registration at [developer.usajobs.gov](https://developer.usajobs.gov/) | U.S. federal government work — generally public domain in the US (17 U.S.C. §105); no separate data license was found on their developer site. |
| **RemoteOK** (`jobs.py`) | Remote job postings, general feed + tag-filtered queries (design/web-dev/general-dev tags) | None | Public API, no published data license found; RemoteOK's own site terms of use apply to redistribution — check [remoteok.com](https://remoteok.com/) before republishing collected postings. |
| **MIT OpenCourseWare** (`jobs.py`, `MITOpenCourseWareScraper`) | Course title, description, department, topics — via MIT's own public aggregation API at `open.mit.edu/api/v0/courses/?offered_by=OCW` (ocw.mit.edu itself is a static site with no REST API; per-course pages only embed Schema.org JSON-LD, which lacks department/topics) | None | **CC BY-NC-SA 4.0** — confirmed directly from the JSON-LD embedded in MIT's own course pages (`"license": "https://creativecommons.org/licenses/by-nc-sa/4.0/"`). Non-commercial, share-alike, attribution required. |
| **Coursera** | **Excluded.** No Coursera dataset file was ever provided or located on the machine this repo was built on, and Coursera has no official public API to collect from independently. Its provenance and license cannot be established, so no Coursera data is included anywhere in this repo — do not assume it's present or attempt to backfill it without a verified, licensed source. | — | N/A — excluded, not "pending." |

The real **course corpus in this repo is MIT OpenCourseWare only** — Coursera
was deliberately left out rather than filled in with unverifiable data.

## Evaluation scope: 6 domains, not 7

The taxonomy defines 7 domains, but **real evaluation/training data (pairs,
gold set) covers only 6**: `computer_science`, `data_science`,
`cloud_devops`, `cybersecurity`, `business`, `design`. `web_development` is
excluded — the MIT OCW course corpus has **zero** courses that classify into
it (verified both by the original topic/department-keyword pass and the
later unified taxonomy-keyword classifier), so there's no course-side data to
pair against `web_development` jobs at all. Job postings tagged
`web_development` still exist in `data/processed/jobs/*.csv`, they just don't
participate in pair generation.

`data/scrapers/coursera.py` remains an unimplemented stub (see its own
docstring) for the same reason: no verifiable public source exists to
implement it against.

## Model modes

- `MODEL_TYPE=bge` (default): zero-shot cosine similarity on the pretrained
  BGE backbone. Works immediately, no training required.
- `MODEL_TYPE=curriculum_gpt`: the seven-component model. Requires a trained
  checkpoint (`MODEL_CHECKPOINT=...`) to be meaningful — with random,
  untrained weights it will not produce useful scores. No checkpoint ships
  with this repo.

## Fine-tuning experiment (2026-07-21) — reverted, zero-shot BGE retained in production

A fine-tuning experiment was run against the production inference path and
**did not ship**: `MODEL_TYPE=bge` with no `MODEL_CHECKPOINT` (zero-shot BGE
backbone cosine) remains the model actually served — this was already
docker-compose's default (`MODEL_CHECKPOINT: ${MODEL_CHECKPOINT:-}`, unset)
and nothing in this repo's config, `.env`, or `checkpoints/` points at a
trained checkpoint. The experiment never touched the persistent config; it
was only ever passed as an ephemeral `MODEL_CHECKPOINT` env var to one-off
evaluation commands.

**What was tried:** `scripts/train_bge_frozen.py`'s frozen-backbone pipeline
(precompute BGE embeddings once, train only the curriculum/job projection
heads via InfoNCE) was extended to accept a graded, per-pair training weight
derived from LLM labels (`aligned`→1.0, `partial`→0.5, `not_aligned`→0.0,
excluded) instead of the binary heuristic `label`. Trained on
`data/labeled/pairs_sample_500_llm.parquet` (500 pairs, fully LLM-labeled;
170 had weight > 0), seed 42, 10 epochs, checkpoint saved to
`outputs/checkpoints/bge_finetuned/`.

**Result: it made every metric worse**, evaluated on the same 300-pair
LLM-graded gold set (`results/gold_evaluation.json` vs.
`results/finetuned_evaluation.json`):

| Metric | Zero-shot BGE | Fine-tuned | Δ |
|---|---|---|---|
| Accuracy | 0.787 | 0.633 | −0.154 |
| AUC | 0.865 | 0.659 | −0.207 |
| F1 | 0.766 | 0.599 | −0.168 |
| NDCG@10 | 0.427 | 0.405 | −0.022 |

Two domains (cloud_devops, design) dropped to near-chance AUC (~0.53). Likely
cause: the training set was too small and imbalanced for InfoNCE-style
contrastive training — only 13 of 170 trainable pairs were full "aligned"
anchors (the rest were half-weighted "partial" pairs), which isn't enough
signal for two small projection heads over 10 epochs without distorting the
embedding space. **Zero-shot BGE is retained as the production model** until
a substantially larger LLM-labeled training set is available (the full
`data/labeled/pairs.parquet` corpus is 4,361 pairs, but LLM-labeling all of
it is still blocked on free-tier LLM API quotas — see the labeling scripts'
own logs for per-key daily token limits encountered during this work).

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
