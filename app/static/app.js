const form = document.getElementById('job-form');
const datasetOptions = document.getElementById('dataset-options');
const modelOptions = document.getElementById('model-options');
const eventFeed = document.getElementById('event-feed');
const overview = document.getElementById('job-overview');
const clearLog = document.getElementById('clear-log');
let currentEventSource = null;

function createCheckbox(name, value, labelText) {
  const label = document.createElement('label');
  label.className = 'check-pill';
  label.innerHTML = `<input type="checkbox" name="${name}" value="${value}" checked /> <span>${labelText}</span>`;
  return label;
}

function addEvent(payload) {
  const item = document.createElement('article');
  item.className = 'feed-item';
  const stamp = new Date().toLocaleTimeString();
  item.innerHTML = `<small>${stamp} · ${payload.event}${payload.dataset ? ` · ${payload.dataset}` : ''}${payload.model ? ` · ${payload.model}` : ''}</small><strong>${payload.message}</strong><pre>${JSON.stringify(payload.data || {}, null, 2)}</pre>`;
  eventFeed.prepend(item);
}

function collectChecked(name) {
  return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((node) => node.value);
}

async function loadConfig() {
  const response = await fetch('/api/config');
  const config = await response.json();
  config.datasets.forEach((dataset) => datasetOptions.appendChild(createCheckbox('datasets', dataset, dataset)));
  config.models.forEach((model) => {
    const label = model.display_name || model.id;
    modelOptions.appendChild(createCheckbox('models', model.id, `${label} (${model.provider})`));
  });
}

function openStream(jobId) {
  if (currentEventSource) currentEventSource.close();
  currentEventSource = new EventSource(`/api/jobs/${jobId}/events`);
  currentEventSource.addEventListener('message', (event) => {
    try {
      addEvent(JSON.parse(event.data));
    } catch (_err) {
      addEvent({ event: 'parse_error', message: event.data, data: {} });
    }
  });
  ['job_created', 'dataset_loaded', 'model_started', 'row_scored', 'model_completed', 'model_failed', 'job_completed', 'job_failed', 'job_cancelled'].forEach((name) => {
    currentEventSource.addEventListener(name, (event) => addEvent(JSON.parse(event.data)));
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const payload = {
    datasets: collectChecked('datasets'),
    models: collectChecked('models'),
    workers: Number(form.workers.value),
    max_samples: Number(form.max_samples.value),
    seed: Number(form.seed.value),
    output_subdir: form.output_subdir.value || null,
    enable_bert_score: form.enable_bert_score.checked,
  };
  const response = await fetch('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  overview.textContent = JSON.stringify(data, null, 2);
  if (!response.ok) {
    addEvent({ event: 'request_error', message: data.detail || 'Failed to create job.', data });
    return;
  }
  addEvent({ event: 'client', message: `Started job ${data.job_id}`, data });
  openStream(data.job_id);
});

clearLog.addEventListener('click', () => {
  eventFeed.innerHTML = '';
});

loadConfig();
