# UniBot Backend

UniBot answers questions about your university using the records you give it.
You provide structured records that you have already collected. UniBot stores
them, builds a search index, and serves answers with citations over an HTTP API.

This repository is the backend only. There is no scraping or data collection
here. You bring your own records.

## How the pipeline works

There are three steps, run in order:

1. Ingest. Read your `data/sources.json` and `data/records.jsonl` and load them
   into the Postgres database.
2. Build. Turn the stored records into a search index (chunk the text, create
   embeddings, write them to Qdrant), then activate that index for serving.
3. Serve. Start the HTTP API. It takes a question, finds the relevant records,
   and returns an answer that cites them.

```
data/sources.json  +  data/records.jsonl
        |
        v
   ingest_records  ->  Postgres
        |
        v
   build_serving_generation  ->  Qdrant index
        |
        v
   API (uvicorn)  ->  answers with citations
```

## What you provide

Put your data in a `data/` folder at the repo root:

```
data/
  sources.json      one entry per source URL or document
  records.jsonl     one record per line (the facts you want answered)
  documents/        optional PDFs or DOCX files referenced by records
```

`sources.json` is a JSON array. Each entry looks like this:

```json
{
  "source_url": "https://your.edu/admissions/",
  "canonical_url": "https://your.edu/admissions",
  "source_class": "admissions_cycle",
  "crawl_method": "html_static",
  "legal_status": "allowed",
  "default_authority_tier": 1,
  "refresh_policy": "weekly",
  "parser_target": "html"
}
```

`records.jsonl` has one record per line. Every record needs these fields:

```json
{"record_id": "university_info:about", "record_type": "university_info", "source_url": "https://your.edu/about/", "source_section_id": "about", "source_section_label": "About", "source_locator": "body", "source_authority_tier": 1, "conflict_scope_id": "university_info:about", "dedupe_key": "university_info:about", "record_payload": {"title": "About", "body_text": "..."}}
```

The exact field definitions are in `partner_kit/`. Copy those dataclasses into
your own data tooling so the records you produce match what the ingester expects.

## Requirements

* Python 3.13 or newer (see `.python-version`).
* uv for installing dependencies. Install with `pip install uv`.
* A PostgreSQL database. You need a direct connection string and a pooled
  connection string. A Neon database works out of the box.
* Qdrant for the vector index. The fastest way to run it locally:
  `docker run -p 6333:6333 qdrant/qdrant`.
* API keys, only if you want production quality answers: a Cohere key for
  embeddings and reranking, and an OpenRouter key for the answering model. You
  do not need keys to run in the default local mode.

## Setup

```bash
# install dependencies
uv sync

# create your config from the template, then edit the values
cp .env.example .env

# create the database tables
uv run alembic upgrade head
```

Open `.env` and set, at minimum, your two Postgres connection strings and the
Qdrant URL. The rest have working defaults. Every setting is documented in
`.env.example`.

## Running it

```bash
# 1. check your records before loading anything (no database writes)
uv run python -m scripts.ingest_records --records data/records.jsonl --dry-run

# 2. load sources and records into the database
uv run python -m scripts.ingest_records --sources data/sources.json --records data/records.jsonl

# 3. build the search index and activate it
uv run python -m scripts.build_serving_generation --generation-label v1

# 4. start the API
uv run uvicorn --factory unibot.api.app:create_app --host 0.0.0.0 --port 8000
```

Run step 1 first and fix anything it reports. It lists records that are missing
required fields or that could not be parsed, and skips them. When the dry run is
clean, run the real ingest in step 2.

A note on high risk records. Admissions cycles, fee schedules, and merit lists
are held back until a second source confirms them. This is on purpose, so a
single unverified source does not produce a wrong answer. Everything else
(programs, faculty, policies, scholarships, general information) is served as
soon as it is ingested.

## Asking questions

```bash
curl -s localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"query": "What programs does the university offer?"}'
```

The response contains the answer and the records it was based on. Questions that
fall outside your data are declined instead of guessed.

## Quality modes

By default the backend runs in a local mode that needs no API keys. It uses
simple embeddings and a deterministic answer builder, which is good for trying
things out but not for production.

For production quality, set `UNIBOT_ENVIRONMENT=production` in `.env` and add
your Cohere and OpenRouter keys. This switches on real embeddings, neural
reranking, and the hosted answering model. See `.env.example` for the full list
of options.

## Customizing

* To make the assistant introduce itself with your university name, edit
  `UNIVERSITY_NAME` near the top of `src/unibot/answering/prompting.py`.
* The URL classifier in `src/unibot/domain/source_policies.py` uses generic
  rules. Your records already carry their own `record_type`, so the classifier
  is only a fallback. Extend it if your URLs follow a specific pattern.

## Project layout

```
src/unibot/
  api/         the HTTP API and routes
  retrieval/   query handling, reranking, filtering
  answering/   prompt building, grounding, citation checks
  indexing/    chunking, embeddings, Qdrant writer, index builds
  pipeline/    record upsert, verification, provenance
  verify/      freshness, deduplication, value identity
  db/          database models, repositories, sessions
  domain/      record types, renderers, source policies
scripts/       commands you run (ingest, build, verify, reset)
alembic/       database migrations
partner_kit/   the record formats to copy into your data tooling
```
