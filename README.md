# CurriculumGPT

CurriculumGPT is a curriculum-to-industry alignment platform. Given a list of
courses/skills — an individual's coursework, or a full institutional
curriculum — it compares them against real job postings using a transformer
embedding model and reports:

- an overall alignment score, industry coverage, and curriculum relevance
- a skill-gap breakdown (which in-demand skills the curriculum doesn't cover)
- best-matching jobs, with matched vs. missing skills per job
- relevant certifications to close the identified gaps
- exportable PDF / Excel reports

The scoring core also implements a seven-component alignment model
(semantic, topic coverage, recency, cognitive depth, skill-graph, trend,
structure) described in the accompanying paper — see
`ml/models/curriculum_gpt.py`. The production model, however, is a simpler
zero-shot BGE embedding model (see [Model](#model) below).

## Installation

```bash
docker compose up --build
```

This starts Postgres, the FastAPI backend (`:7000`), and the React frontend
(`:5173`). On first boot the backend downloads the pretrained
`BAAI/bge-large-en-v1.5` embedding model (~1.3GB) from Hugging Face — this
needs internet access and takes a few minutes. The database seeds itself
automatically from the sample data under `data/mappings/` and
`data/processed/` (see [Data](#data) below).

Open `http://localhost:5173` once it's up.

No `.env` file is required to get started — `docker-compose.yml` has working
defaults for everything. Copy `.env.example` to `.env` if you want to add
real API keys (O\*NET, USAJobs, Groq/Gemini) for the data-collection and
LLM-labeling scripts. **Never commit `.env`** — it's gitignored, and
`.env.example` should only ever contain placeholder values.

## Project structure

```
backend/    FastAPI app: REST API, DB models, alignment service
frontend/   React + Vite + Tailwind UI
ml/         Models (BGE bi-encoder, 7-component CurriculumGPT), training, evaluation
data/       Skill taxonomy, certifications, scrapers, preprocessing, and courses/jobs data
scripts/    CLI entry points for scraping, preprocessing, dataset building, training, evaluation
results/    Evaluation JSONs and report figures (see results/figures/)
```

## Data

### Seed data (ships with the repo)

`data/mappings/*.json` and `data/processed/{courses,jobs}/sample_*.csv` are
**hand-authored sample data**, not scraped from live sources — enough to
demonstrate the full pipeline end-to-end (55 skills, 52 sample courses, 52
sample jobs across IT, engineering, finance, marketing, operations, HR, and
design) without needing any API keys.

### Real data collection (`data/scrapers/`)

Beyond the seed data, `data/scrapers/` implements real collectors against
genuine public sources. Output lands in `data/raw/{courses,jobs}/*.jsonl`;
`data/preprocessing/pipeline.py` then cleans, language-filters, deduplicates,
and flattens metadata into `data/processed/{courses,jobs}/*.csv`.

| Source | What it collects | Auth needed | License / terms |
|---|---|---|---|
| **O\*NET Web Services** (`onet.py`) | Occupations + skills + technology skills, one query per taxonomy domain, deduped by O\*NET-SOC code | `ONET_API_KEY` — free registration at [services.onetcenter.org](https://services.onetcenter.org/). Current accounts use the v2.0 API at `api-v2.onetcenter.org` with an `X-API-Key` header — the legacy `services.onetcenter.org/ws/` host uses old-style Basic Auth and is a separate system. | See [services.onetcenter.org/help/license_data](https://services.onetcenter.org/help/license_data) — O\*NET data is sponsored by the U.S. Department of Labor; check that page for current attribution/usage terms before redistributing collected data. |
| **USAJobs** (`jobs.py`) | Federal job postings, keyword-swept per taxonomy domain | `USAJOBS_API_KEY` + `USAJOBS_EMAIL` — free registration at [developer.usajobs.gov](https://developer.usajobs.gov/) | U.S. federal government work — generally public domain in the US (17 U.S.C. §105); no separate data license found on their developer site. |
| **RemoteOK** (`jobs.py`) | Remote job postings, general feed + tag-filtered queries | None | Public API, no published data license; RemoteOK's site terms of use apply to redistribution — check [remoteok.com](https://remoteok.com/) before republishing collected postings. |
| **MIT OpenCourseWare** (`jobs.py`, `MITOpenCourseWareScraper`) | Course title, description, department, topics — via MIT's public aggregation API at `open.mit.edu/api/v0/courses/?offered_by=OCW` | None | **CC BY-NC-SA 4.0** — confirmed from the JSON-LD embedded in MIT's own course pages. Non-commercial, share-alike, attribution required. |
| **Coursera** | **Excluded.** No Coursera dataset or public API was ever available to this project; provenance and license can't be established, so no Coursera data is included anywhere in this repo. | — | N/A — excluded, not "pending." `data/scrapers/coursera.py` is an unimplemented stub. |

The real **course corpus in this repo is MIT OpenCourseWare only**.

### Evaluation scope: 6 domains

The taxonomy defines 7 domains, but real evaluation/training data (pairs,
gold set) covers only **6**: `computer_science`, `data_science`,
`cloud_devops`, `cybersecurity`, `business`, `design`. `web_development` is
excluded — the MIT OCW course corpus has essentially zero courses that
classify into it, so there's no course-side data to pair against
`web_development` jobs. Job postings tagged `web_development` still exist in
`data/processed/jobs/*.csv`; they just don't participate in pair generation.

## Reproducing the pipeline

All commands below assume the stack is running (`docker compose up -d`) and
are run via `docker compose exec backend <command>`. Each stage's output
feeds the next.

```bash
# 1. Collect real data (needs ONET_API_KEY, USAJOBS_API_KEY/EMAIL in .env)
docker compose exec backend python scripts/scrape.py --source all

# 2. Clean, dedupe, and flatten into data/processed/{courses,jobs}/*.csv
docker compose exec backend python scripts/preprocess.py

# 3. Build labeled pairs: BGE similarity retrieval, fixed-threshold labeling,
#    hard-negative mining, and a disjoint stratified gold holdout
docker compose exec backend python scripts/build_dataset.py \
  --pairs-per-course 10 --hard-neg-per-domain 40 --gold-size 300 --seed 42

# 4. (Optional) Grade pairs with an LLM for higher-quality ground truth —
#    needs GROQ_API_KEY (and optionally GROQ_API_KEY_2 as a same-day fallback)
#    or GEMINI_API_KEY in .env
docker compose exec backend python scripts/build_dataset.py --relabel-graded \
  --provider groq --input data/labeled/gold.parquet \
  --out data/labeled/gold_llm.parquet --checkpoint-every 50

# 5. Evaluate the production model against LLM-graded ground truth
docker compose exec backend python scripts/evaluate_gold.py \
  --input data/labeled/gold_llm.parquet --out results/gold_evaluation.json

# 6. Regenerate the report figures (results/figures/*.png, 300 DPI)
docker compose exec backend python scripts/generate_figures.py
```

`scripts/train_bge_frozen.py` (frozen-backbone InfoNCE fine-tuning of the BGE
projection heads) is also available but is **not** part of the recommended
pipeline — see [Model](#model) for why.

## Model

- `MODEL_TYPE=bge` (default, production): zero-shot cosine similarity on the
  pretrained BGE backbone. Works immediately, no training required.
- `MODEL_TYPE=curriculum_gpt`: the seven-component model. Requires a trained
  checkpoint (`MODEL_CHECKPOINT=...`) to be meaningful — with random,
  untrained weights it will not produce useful scores. No checkpoint ships
  with this repo.


```
