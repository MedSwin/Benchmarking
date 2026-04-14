# Operations Manual

## 1. Environment variables

### Core variables

- `DATA_ROOT`: root directory containing `medquad/`, `medmcqa/`, and `healthbench/`.
- `OUTPUT_ROOT`: directory where benchmark jobs will be written.
- `DEFAULT_WORKERS`: default concurrency if the UI/request does not override it.
- `REQUEST_TIMEOUT_SECONDS`: timeout for outbound LLM API calls.

### API key lists

Each provider accepts a comma-separated key list:

- `OPENAI_API_KEYS`
- `GOOGLE_API_KEYS`
- `XAI_API_KEYS`
- `MISTRAL_API_KEYS`

The app rotates across keys in round-robin order. Each key has its own limiter. This means one exhausted key does not block another still under quota.

### Rate limits

Per-provider request budgets are configured with:

- `OPENAI_REQUESTS_PER_MINUTE`
- `GOOGLE_REQUESTS_PER_MINUTE`
- `XAI_REQUESTS_PER_MINUTE`
- `MISTRAL_REQUESTS_PER_MINUTE`

When a worker asks for a request slot and the current key has exhausted its 60-second window, the limiter pauses until the oldest request falls out of the window.

## 2. Dataset expectations

### MedQuAD

The loader expects JSONL rows containing at least:

- `id`
- `question`
- `answer`

### MedMCQA

The loader expects JSONL rows containing at least:

- `question`
- `opa`, `opb`, `opc`, `opd`
- one answer field such as `cop`, `answer`, `answers`, `correct_option`, or `correct_options`
- optional `choice_type`, `subject_name`, `topic_name`

### HealthBench

The loader accepts the same flexible layouts handled by the old script, including prompts/messages stored in:

- `prompt`
- `messages`
- `processed_prompt_en_plaintext`
- `prompt_text`

And references stored in:

- `ideal_completions_data.ideal_completion`
- `ideal_completions_data.ideal_completions_ref_completions[*]`
- fallback flat fields like `ideal_completion`, `answer`, `reference`, or `gold_answer`

## 3. Operational flow

1. Start the server with `uvicorn app.main:app --reload`.
2. Open the dashboard in a browser.
3. Launch a smoke test first, typically `max_samples=10`.
4. Confirm logs, event flow, and artifact creation under `output/<job_id>/`.
5. Scale up to the full dataset once quotas and latency are understood.

## 4. Auditing and troubleshooting

### Terminal logging

The app emits structured operational messages to the terminal using Python logging.

### File-based auditing

Every job writes:

- `audit.log.jsonl`: structured log stream.
- `events.jsonl`: UI/event stream payloads.
- per-model `.audit.jsonl` rows.

### Common failures

- **Git LFS pointer detected**: run `git lfs pull`.
- **Provider key missing**: populate the corresponding API key environment variable.
- **429 / quota errors**: lower worker count, add more keys, or reduce the provider RPM setting to better match the contract of your account.
- **High BERTScore runtime**: disable BERTScore in the UI for quick smoke tests.

## 5. Extending providers

Provider integrations live in `app/providers.py`. To add another hosted model:

1. Add the model enum to `app/models.py`.
2. Map it to a provider in `MODEL_PROVIDER`.
3. Implement the outbound request method in `ProviderPool`.
4. Surface the model in `/api/config`.

## 6. Running without the UI

You can also submit jobs directly:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "datasets": ["medquad", "medmcqa"],
    "models": ["gpt-5.1", "grok-4-1-fast-reasoning"],
    "workers": 10,
    "max_samples": 100,
    "seed": 13,
    "enable_bert_score": true
  }'
```
