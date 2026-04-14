# Medical Benchmark Control Plane

This repository now includes a FastAPI application under `app/` that reproduces the benchmark logic from the three original single-file scripts while targeting four hosted models instead of local Hugging Face checkpoints: Gemini 3.1 Pro Preview, GPT-5.1, Grok 4.1 Fast Reasoning, and Mistral Large 3.

## What the app preserves from the original scripts

The new app intentionally keeps the same dataset-specific benchmark behavior:

- **MedQuAD**: same prompt framing and the same five overlap-style metrics derived from the reference answer and model answer: Rouge-L F1, BERTScore F1, token F1, unigram precision, and bigram precision.
- **HealthBench**: same canonical-reference extraction strategy as the prior script, including fallback lookup order for `ideal_completion`-style fields, and the same five metrics as MedQuAD.
- **MedMCQA**: same question + four-option layout, same strict answer parsing logic, and equivalent correctness/validity auditing. To keep the app’s cross-dataset dashboard consistent, MedMCQA exposes the same five metric fields, with exact-match correctness projected into those metric columns and the detailed categorical audit fields preserved alongside them.

## Repository layout

- `data/`: expected location of the three benchmark datasets.
- `script/`: preserved legacy benchmark scripts for reference.
- `app/`: FastAPI backend, provider adapters, dataset loaders, metrics, live event streaming, and frontend assets.
- `tests/`: unit tests for loaders, scoring helpers, and MedMCQA parsing behavior.

## Important dataset note

The current repository stores the dataset files as **Git LFS pointers**. Before running the app, fetch the actual payloads:

```bash
git lfs pull
```

If you skip that step, the app will refuse to start a benchmark against pointer files so you do not accidentally benchmark against the wrong input.

## Setup

### 1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2. Configure the `.env`

Copy the template and populate it with one or more API keys per provider:

```bash
cp .env.example .env
```

The load balancer accepts comma-separated key lists like this:

```dotenv
OPENAI_API_KEYS=key-1,key-2,key-3
GOOGLE_API_KEYS=key-1,key-2
XAI_API_KEYS=key-1
MISTRAL_API_KEYS=key-1,key-2
```

Each provider/key pair gets its own rate limiter. When one key reaches its per-minute limit, the app waits for availability and then continues, which satisfies the “await the API call limit once reached” requirement without dropping work.

## Running the app

```bash
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000`.

## Benchmark workflow

1. Select one or more datasets.
2. Select one or more target models.
3. Set `workers` to the desired concurrency level. The default is `10`.
4. Optionally cap `max_samples` for a smoke test.
5. Launch the job and watch the live event stream.

The backend schedules row-level work through `asyncio`, fans requests out across the worker pool, and rotates across available API keys per provider. Output is written to `output/<job_id>/...`.

## Output artifacts

For each dataset/model combination, the app writes:

- `<model>.csv`: row-level benchmark details.
- `<model>.audit.jsonl`: row-level audit trail.
- `<model>.metrics.json`: metric means and artifact references.
- `summary.json`: dataset-level summary.
- `audit.log.jsonl` / `events.jsonl` at the job root: terminal + UI event audit stream.

## API endpoints

- `GET /`: frontend dashboard.
- `GET /api/config`: available datasets/models for the UI.
- `POST /api/jobs`: create a new benchmark job.
- `GET /api/jobs/{job_id}`: inspect current status.
- `DELETE /api/jobs/{job_id}`: request cancellation.
- `GET /api/jobs/{job_id}/events`: Server-Sent Events stream for live progress.

## Architecture summary

### Backend

- `app/datasets.py` mirrors the dataset-specific prompt and reference extraction logic from the legacy scripts.
- `app/providers.py` contains provider adapters for OpenAI, Google Gemini, xAI, and Mistral.
- `app/rate_limit.py` adds a per-key asynchronous rate limiter.
- `app/runner.py` manages jobs, concurrency, scoring, persistence, and event emission.
- `app/audit.py` records structured logs and event payloads for every run.

### Frontend

The frontend is intentionally lightweight: plain HTML, CSS, and JavaScript served by FastAPI. It provides a clear operator view for job launch, live event streaming, and job-status inspection without introducing a separate Node build step.

## Notes on the four target models

The app defaults are wired for:

- `gemini-3.1-pro-preview`
- `gpt-5.1`
- `grok-4-1-fast-reasoning`
- `mistral-large-latest`

If your provider account exposes a different alias, override the adapter or endpoint configuration in `.env` or `app/main.py` as needed.

## Testing

Run the included tests with:

```bash
pytest
```

## Containerization

Build the production image once and start the service with Docker Compose so datasets stay on the host:

```bash
docker compose up --build
```

If you prefer plain Docker, install the image, bind-mount `./data` (read-only) and an `output` volume, and pass `.env` manually:

```bash
docker build -t benchmarking-app:latest .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/data:/app/data:ro" \
  -v benchapp-output:/app/output \
  --env-file .env \
  benchmarking-app:latest
```

The Compose setup already binds `./data` for datasets, writes outputs to the `output` volume, and exposes port 8000 with a health-check on `/api/config`. Make sure to run `git lfs pull` before starting any container so the datasets are the actual payloads.
