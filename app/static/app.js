const form = document.getElementById('job-form');
const datasetOptions = document.getElementById('dataset-options');
const modelOptions = document.getElementById('model-options');
const eventFeed = document.getElementById('event-feed');
const overview = document.getElementById('job-overview');
const clearLog = document.getElementById('clear-log');
const metricsTableBody = document.querySelector('#metrics-table tbody');
const streamStatus = document.getElementById('stream-status');
const jobStatus = document.getElementById('job-status');
const rowsScored = document.getElementById('rows-scored');
const lastEventTime = document.getElementById('last-event-time');
const launchButton = document.getElementById('launch-btn');

const metricKeys = ['rougeL_f', 'tok_f1', 'uni_prec', 'bi_prec', 'bert_f'];
const FINAL_STATUSES = new Set(['completed', 'failed', 'cancelled']);
const LAST_JOB_KEY = 'benchapp:last-job-id';

let currentEventSource = null;
let metricsState = {};
let modelLookup = {};
let totalRowsScored = 0;
let refreshRow = 10;

// Motivation vs Logic: Persisting the latest job ID lets us reconnect to its stream after a refresh so the active run stays visible.
function persistJobId(jobId) {
  try {
    if (jobId) {
      localStorage.setItem(LAST_JOB_KEY, jobId);
    } else {
      localStorage.removeItem(LAST_JOB_KEY);
    }
  } catch (_err) {
    /* ignore storage failures */
  }
}

function readPersistedJobId() {
  try {
    return localStorage.getItem(LAST_JOB_KEY);
  } catch (_err) {
    return null;
  }
}

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
  if (lastEventTime) {
    lastEventTime.textContent = stamp;
  }
}

function collectChecked(name) {
  return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((node) => node.value);
}

function formatMetric(value) {
  if (value == null || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(3);
}

function setOverview(job) {
  if (!job) {
    overview.textContent = 'No job yet.';
    setJobBadge('No job', 'idle');
    return;
  }
  overview.textContent = JSON.stringify(job, null, 2);
  setJobBadge(job.status || 'queued', FINAL_STATUSES.has(job.status) ? 'done' : 'live');
}

function resetMetricsPlaceholder() {
  metricsState = {};
  metricsTableBody.innerHTML = '<tr class="empty-row"><td colspan="7">Select models and run a benchmark to unlock live metrics.</td></tr>';
}

function setBadge(element, text, tone = 'idle') {
  if (!element) {
    return;
  }
  element.textContent = text;
  element.classList.remove('status-live', 'status-idle', 'status-error', 'status-done');
  if (tone === 'live') {
    element.classList.add('status-live');
  } else if (tone === 'error') {
    element.classList.add('status-error');
  } else if (tone === 'done') {
    element.classList.add('status-done');
  } else {
    element.classList.add('status-idle');
  }
}

function setStreamBadge(state) {
  if (state === 'connected') {
    setBadge(streamStatus, 'Live', 'live');
  } else if (state === 'error') {
    setBadge(streamStatus, 'Reconnecting', 'error');
  } else if (state === 'closed') {
    setBadge(streamStatus, 'Completed', 'done');
  } else {
    setBadge(streamStatus, 'Idle', 'idle');
  }
}

function setJobBadge(statusText, tone = 'idle') {
  setBadge(jobStatus, statusText, tone);
}

function setLaunchPending(isPending) {
  if (!launchButton) {
    return;
  }
  launchButton.disabled = isPending;
  launchButton.textContent = isPending ? 'Launching...' : 'Launch benchmark';
}

function resetLiveStats() {
  totalRowsScored = 0;
  if (rowsScored) {
    rowsScored.textContent = '0';
  }
  if (lastEventTime) {
    lastEventTime.textContent = '—';
  }
}

function updateLiveStatusFromEvent(payload) {
  if (!payload?.event) {
    return;
  }
  if (payload.event === 'row_scored') {
    totalRowsScored += 1;
    if (rowsScored) {
      rowsScored.textContent = String(totalRowsScored);
    }
  }
  if (payload.event.endsWith('failed')) {
    setJobBadge('Failed', 'error');
  } else if (payload.event.endsWith('completed')) {
    setJobBadge('Completed', 'done');
  } else if (payload.event.endsWith('cancelled')) {
    setJobBadge('Cancelled', 'done');
  } else if (payload.event === 'job_created' || payload.event === 'model_started') {
    setJobBadge('Running', 'live');
  }
  if (payload.event.startsWith('job_')) {
    flushAllPendingMetrics();
  }
  if (payload.event.startsWith('model_')) {
    flushPendingMetrics(payload.model);
  }
}

function buildMetricsTable(modelIds) {
  const normalized = [...new Set((modelIds || []).map((value) => value && value.toString()))].filter(Boolean);
  if (!normalized.length) {
    resetMetricsPlaceholder();
    return;
  }
  metricsState = {};
  metricsTableBody.innerHTML = '';
  normalized.forEach((modelId) => {
    const meta = modelLookup[modelId] || { display_name: modelId, provider: '' };
    const label = meta.display_name || modelId;
    const providerHint = meta.provider ? `<span class="model-provider">${meta.provider}</span>` : '';
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>
        <div class="model-name">
          <strong>${label}</strong>
          ${providerHint}
        </div>
      </td>
      <td data-col="dataset">—</td>
      ${metricKeys.map((key) => `<td data-col="${key}" class="numeric">—</td>`).join('')}
    `;
    metricsTableBody.appendChild(row);
    const totals = {};
    metricKeys.forEach((key) => {
      totals[key] = 0;
    });
    metricsState[label] = {
      rowElement: row,
      modelId,
      dataset: null,
      count: 0,
      totals,
      pendingSinceRender: 0,
    };
    metricsState[modelId] = metricsState[label];
  });
}

function getNormalizedModelId(model) {
  if (!model) {
    return '';
  }
  if (typeof model === 'string') {
    return model;
  }
  if (typeof model === 'object') {
    if ('id' in model && model.id) {
      return model.id;
    }
    if ('value' in model && model.value) {
      return model.value;
    }
  }
  return String(model);
}

// Motivation vs Logic: Aggregating row metrics keeps the new table in sync with the SSE stream without reloading.
function updateMetricsFromRow(modelLabel, dataset, data) {
  if (!modelLabel) {
    return;
  }
  const state = metricsState[modelLabel];
  if (!state) {
    return;
  }
  state.count += 1;
  if (dataset) {
    state.dataset = dataset;
  }
  metricKeys.forEach((key) => {
    const raw = data?.[key];
    const numeric = typeof raw === 'number' ? raw : Number(raw);
    state.totals[key] += Number.isFinite(numeric) ? numeric : 0;
  });
  state.pendingSinceRender += 1;
  if (state.pendingSinceRender >= refreshRow) {
    renderMetricsRow(state);
    state.pendingSinceRender = 0;
  }
}

function flushPendingMetrics(modelLabel) {
  if (!modelLabel) {
    return;
  }
  const state = metricsState[modelLabel];
  if (!state || !state.pendingSinceRender) {
    return;
  }
  renderMetricsRow(state);
  state.pendingSinceRender = 0;
}

function flushAllPendingMetrics() {
  const seen = new Set();
  Object.values(metricsState).forEach((state) => {
    if (!state || !state.pendingSinceRender) {
      return;
    }
    if (seen.has(state)) {
      return;
    }
    seen.add(state);
    renderMetricsRow(state);
    state.pendingSinceRender = 0;
  });
}

function renderMetricsRow(state) {
  const { rowElement, dataset, count, totals } = state;
  const datasetCell = rowElement.querySelector('[data-col="dataset"]');
  if (datasetCell) {
    datasetCell.textContent = dataset || '—';
  }
  metricKeys.forEach((key) => {
    const cell = rowElement.querySelector(`[data-col="${key}"]`);
    if (cell) {
      const average = count ? totals[key] / count : 0;
      cell.textContent = count ? formatMetric(average) : '—';
    }
  });
  rowElement.classList.remove('row-live-pulse');
  // Motivation vs Logic: Subtle pulse animation gives instant visual confirmation that live scoring updated a row.
  requestAnimationFrame(() => rowElement.classList.add('row-live-pulse'));
}

function openStream(jobId) {
  if (currentEventSource) {
    currentEventSource.close();
  }
  setStreamBadge('idle');
  currentEventSource = new EventSource(`/api/jobs/${jobId}/events`);
  currentEventSource.onopen = () => {
    setStreamBadge('connected');
  };
  currentEventSource.onerror = () => {
    setStreamBadge('error');
  };
  currentEventSource.addEventListener('message', (event) => {
    try {
      const payload = JSON.parse(event.data);
      // Named SSE events are handled by dedicated listeners below; this is only fallback for plain messages.
      if (payload?.event) {
        return;
      }
      addEvent({
        event: 'message',
        message: payload?.message || 'Received stream message',
        data: payload || {},
      });
    } catch (_err) {
      addEvent({ event: 'parse_error', message: event.data, data: {} });
    }
  });
  ['job_created', 'dataset_loaded', 'model_started', 'row_generated', 'row_scored', 'model_completed', 'model_failed', 'job_completed', 'job_failed', 'job_cancelled'].forEach((name) => {
    currentEventSource.addEventListener(name, (event) => {
      try {
        const payload = JSON.parse(event.data);
        addEvent(payload);
        updateLiveStatusFromEvent(payload);
        if (payload.event === 'row_scored') {
          updateMetricsFromRow(payload.model, payload.dataset, payload.data);
        }
        if (FINAL_STATUSES.has(payload.event.replace('job_', ''))) {
          setStreamBadge('closed');
          currentEventSource.close();
        }
      } catch (_err) {
        addEvent({ event: 'parse_error', message: event.data, data: {} });
      }
    });
  });
}

async function loadConfig() {
  const response = await fetch('/api/config');
  const config = await response.json();
  const configuredRefresh = Number(config.refresh_row);
  refreshRow = Number.isFinite(configuredRefresh) && configuredRefresh > 0 ? Math.floor(configuredRefresh) : 10;
  modelLookup = config.models.reduce((acc, model) => {
    acc[model.id] = model;
    return acc;
  }, {});
  config.datasets.forEach((dataset) => datasetOptions.appendChild(createCheckbox('datasets', dataset, dataset)));
  config.models.forEach((model) => {
    const label = model.display_name || model.id;
    modelOptions.appendChild(createCheckbox('models', model.id, `${label} (${model.provider})`));
  });
}

async function resumeLastJob() {
  const jobId = readPersistedJobId();
  if (!jobId) {
    buildMetricsTable([]);
    resetLiveStats();
    setStreamBadge('idle');
    setJobBadge('No job', 'idle');
    return;
  }
  try {
    const response = await fetch(`/api/jobs/${jobId}`);
    if (!response.ok) {
      persistJobId(null);
      buildMetricsTable([]);
      resetLiveStats();
      setStreamBadge('idle');
      setJobBadge('No job', 'idle');
      return;
    }
    const job = await response.json();
    setOverview(job);
    const selectedModels = (job.request?.models ?? []).map(getNormalizedModelId);
    buildMetricsTable(selectedModels);
    resetLiveStats();
    if (!FINAL_STATUSES.has(job.status)) {
      setJobBadge('Running', 'live');
      openStream(jobId);
    } else {
      setStreamBadge('closed');
      addEvent({ event: 'client', message: `Loaded last job ${jobId} (${job.status})`, data: job });
    }
  } catch (error) {
    console.warn('Unable to resume last job', error);
    persistJobId(null);
    buildMetricsTable([]);
    resetLiveStats();
    setStreamBadge('error');
    setJobBadge('Unavailable', 'error');
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  setLaunchPending(true);
  try {
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
    if (!response.ok) {
      overview.textContent = JSON.stringify(data, null, 2);
      setJobBadge('Request failed', 'error');
      addEvent({ event: 'request_error', message: data.detail || 'Failed to create job.', data });
      return;
    }
    resetLiveStats();
    setOverview(data);
    persistJobId(data.job_id);
    buildMetricsTable(payload.models);
    setJobBadge('Running', 'live');
    addEvent({ event: 'client', message: `Started job ${data.job_id}`, data });
    openStream(data.job_id);
  } catch (error) {
    setJobBadge('Request failed', 'error');
    addEvent({ event: 'request_error', message: 'Network error while creating job.', data: { error: String(error) } });
  } finally {
    setLaunchPending(false);
  }
});

clearLog.addEventListener('click', () => {
  eventFeed.innerHTML = '';
});

async function init() {
  await loadConfig();
  await resumeLastJob();
}

init();
