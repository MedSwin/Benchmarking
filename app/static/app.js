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
const continueButton = document.getElementById('continue-btn');
const pauseButton = document.getElementById('pause-btn');
const sessionSelect = document.getElementById('session-select');

const metricKeys = ['rougeL_f', 'tok_f1', 'uni_prec', 'bi_prec', 'bert_f'];
const TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled', 'paused']);
const RESUMABLE_STATUSES = new Set(['failed', 'cancelled', 'paused']);
const ACTIVE_STATUSES = new Set(['queued', 'running']);
const LAST_JOB_KEY = 'benchapp:last-job-id';

let currentEventSource = null;
let metricsState = {};
let modelLookup = {};
let totalRowsScored = 0;
let refreshRow = 10;
let activeJobId = null;
let currentJob = null;
let pendingAction = null;

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

function clearEventFeed() {
  eventFeed.innerHTML = '';
}

function formatTimestamp(rawTimestamp) {
  if (!rawTimestamp) {
    return new Date().toLocaleTimeString();
  }
  const parsed = new Date(rawTimestamp);
  if (Number.isNaN(parsed.getTime())) {
    return String(rawTimestamp);
  }
  return parsed.toLocaleString();
}

function addEvent(payload) {
  const item = document.createElement('article');
  item.className = 'feed-item';
  const stamp = formatTimestamp(payload?.ts);
  item.innerHTML = `<small>${stamp} · ${payload.event}${payload.dataset ? ` · ${payload.dataset}` : ''}${payload.model ? ` · ${payload.model}` : ''}</small><strong>${payload.message || 'Event received'}</strong><pre>${JSON.stringify(payload.data || {}, null, 2)}</pre>`;
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
  } else if (state === 'paused') {
    setBadge(streamStatus, 'Paused', 'idle');
  } else if (state === 'error') {
    setBadge(streamStatus, 'Reconnecting', 'error');
  } else if (state === 'closed') {
    setBadge(streamStatus, 'Completed', 'done');
  } else {
    setBadge(streamStatus, 'Idle', 'idle');
  }
}

function setJobBadge(text, tone = 'idle') {
  setBadge(jobStatus, text, tone);
}

function setOverview(job) {
  currentJob = job || null;
  if (!job) {
    overview.textContent = 'No job yet.';
    setJobBadge('No job', 'idle');
    updateControlButtons();
    return;
  }
  overview.textContent = JSON.stringify(job, null, 2);
  if (job.status === 'failed') {
    setJobBadge(job.status, 'error');
  } else if (job.status === 'paused') {
    setJobBadge(job.status, 'idle');
  } else if (TERMINAL_STATUSES.has(job.status)) {
    setJobBadge(job.status, 'done');
  } else {
    setJobBadge(job.status || 'queued', 'live');
  }
  updateControlButtons();
}

function updateControlButtons() {
  if (!launchButton || !continueButton || !pauseButton) {
    return;
  }
  launchButton.disabled = pendingAction !== null;
  continueButton.disabled = pendingAction !== null || !currentJob || !RESUMABLE_STATUSES.has(currentJob.status);
  pauseButton.disabled = pendingAction !== null || !currentJob || !ACTIVE_STATUSES.has(currentJob.status);
}

function setPendingAction(nextAction = null) {
  pendingAction = nextAction;
  if (launchButton) {
    launchButton.textContent = nextAction === 'launch' ? 'Launching...' : 'Launch';
  }
  if (continueButton) {
    continueButton.textContent = nextAction === 'continue' ? 'Continuing...' : 'Continue';
  }
  if (pauseButton) {
    pauseButton.textContent = nextAction === 'pause' ? 'Pausing...' : 'Pause';
  }
  updateControlButtons();
}

function mergeCurrentJob(patch) {
  if (!currentJob) {
    return;
  }
  currentJob = { ...currentJob, ...patch };
  overview.textContent = JSON.stringify(currentJob, null, 2);
  if (activeJobId && currentJob.job_id === activeJobId) {
    if (currentJob.status === 'failed') {
      setJobBadge('failed', 'error');
    } else if (currentJob.status === 'paused') {
      setJobBadge('paused', 'idle');
    } else if (TERMINAL_STATUSES.has(currentJob.status)) {
      setJobBadge(currentJob.status, 'done');
    } else {
      setJobBadge(currentJob.status || 'queued', 'live');
    }
  }
  updateControlButtons();
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

function closeStream() {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
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

function getModelMeta(modelId) {
  return modelLookup[modelId] || { id: modelId, display_name: modelId, provider: '' };
}

function resolveModelId(modelRef) {
  const normalized = getNormalizedModelId(modelRef);
  if (normalized && modelLookup[normalized]) {
    return normalized;
  }
  const byDisplayName = Object.values(modelLookup).find((model) => model.display_name === normalized);
  if (byDisplayName) {
    return byDisplayName.id;
  }
  return normalized;
}

function makeMetricsKey(dataset, modelId) {
  return `${dataset || 'unknown'}::${modelId || 'unknown'}`;
}

function resetMetricsPlaceholder() {
  metricsState = {};
  metricsTableBody.innerHTML = '<tr class="empty-row"><td colspan="7">Select a session or run a benchmark to populate metrics.</td></tr>';
}

// Motivation vs Logic:
// Motivation: restored sessions can span multiple datasets and models, so a single row per model drops data.
// Logic: render one metrics row per dataset/model pair, then hydrate those rows from persisted history and summaries.
function buildMetricsTable(datasetIds, modelIds) {
  const datasets = [...new Set((datasetIds || []).map((value) => String(value || '').trim()).filter(Boolean))];
  const models = [...new Set((modelIds || []).map((value) => resolveModelId(value)).filter(Boolean))];
  if (!datasets.length || !models.length) {
    resetMetricsPlaceholder();
    return;
  }

  metricsState = {};
  metricsTableBody.innerHTML = '';

  datasets.forEach((dataset) => {
    models.forEach((modelId) => {
      const meta = getModelMeta(modelId);
      const providerHint = meta.provider ? `<span class="model-provider">${meta.provider}</span>` : '';
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${dataset}</td>
        <td>
          <div class="model-name">
            <strong>${meta.display_name || modelId}</strong>
            ${providerHint}
          </div>
        </td>
        ${metricKeys.map((key) => `<td data-col="${key}" class="numeric">—</td>`).join('')}
      `;
      metricsTableBody.appendChild(row);

      const totals = {};
      metricKeys.forEach((key) => {
        totals[key] = 0;
      });

      metricsState[makeMetricsKey(dataset, modelId)] = {
        rowElement: row,
        dataset,
        modelId,
        count: 0,
        totals,
        pendingSinceRender: 0,
        summaryMetrics: null,
      };
    });
  });
}

function renderMetricsRow(state) {
  metricKeys.forEach((key) => {
    const cell = state.rowElement.querySelector(`[data-col="${key}"]`);
    if (!cell) {
      return;
    }
    const summaryValue = state.summaryMetrics ? state.summaryMetrics[key] : null;
    const average = summaryValue != null ? Number(summaryValue) : (state.count ? state.totals[key] / state.count : null);
    cell.textContent = average == null ? '—' : formatMetric(average);
  });
  state.rowElement.classList.remove('row-live-pulse');
  requestAnimationFrame(() => state.rowElement.classList.add('row-live-pulse'));
}

function updateMetricsFromRow(modelRef, dataset, data) {
  const modelId = resolveModelId(modelRef);
  const state = metricsState[makeMetricsKey(dataset, modelId)];
  if (!state) {
    return;
  }
  state.summaryMetrics = null;
  state.count += 1;
  metricKeys.forEach((key) => {
    const numeric = Number(data?.[key]);
    state.totals[key] += Number.isFinite(numeric) ? numeric : 0;
  });
  state.pendingSinceRender += 1;
  if (state.pendingSinceRender >= refreshRow) {
    renderMetricsRow(state);
    state.pendingSinceRender = 0;
  }
}

function flushPendingMetrics(modelRef, dataset = null) {
  if (modelRef && dataset) {
    const state = metricsState[makeMetricsKey(dataset, resolveModelId(modelRef))];
    if (state?.pendingSinceRender) {
      renderMetricsRow(state);
      state.pendingSinceRender = 0;
    }
    return;
  }
  Object.values(metricsState).forEach((state) => {
    if (!state.pendingSinceRender) {
      return;
    }
    renderMetricsRow(state);
    state.pendingSinceRender = 0;
  });
}

function applySummaryMetrics(job) {
  Object.entries(job?.datasets || {}).forEach(([datasetName, datasetSummary]) => {
    Object.values(datasetSummary?.models || {}).forEach((modelSummary) => {
      const modelId = resolveModelId(modelSummary?.model_id || modelSummary?.display_name);
      const state = metricsState[makeMetricsKey(datasetName, modelId)];
      if (!state) {
        return;
      }
      state.count = Number(modelSummary?.rows || 0);
      state.summaryMetrics = modelSummary?.metric_means || {};
      renderMetricsRow(state);
      state.pendingSinceRender = 0;
    });
  });
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
  if (payload.event === 'job_failed') {
    mergeCurrentJob({ status: 'failed', finished_at: payload.ts || currentJob?.finished_at, error: payload.message || currentJob?.error });
  } else if (payload.event === 'job_completed' || payload.event === 'job_cancelled') {
    mergeCurrentJob({ status: payload.event.replace('job_', ''), finished_at: payload.ts || currentJob?.finished_at });
  } else if (payload.event === 'job_paused') {
    mergeCurrentJob({ status: 'paused', finished_at: payload.ts || currentJob?.finished_at, error: payload.message || currentJob?.error });
  } else if (payload.event === 'job_created' || payload.event === 'job_resumed' || payload.event === 'model_started') {
    mergeCurrentJob({ status: 'running', finished_at: null, error: null });
  }
  if (payload.event === 'row_scored') {
    updateMetricsFromRow(payload.model, payload.dataset, payload.data);
  }
  if (payload.event.startsWith('model_')) {
    flushPendingMetrics(payload.model, payload.dataset);
  }
  if (payload.event.startsWith('job_')) {
    flushPendingMetrics();
  }
}

function replayHistory(events) {
  (events || []).forEach((payload) => {
    addEvent(payload);
    updateLiveStatusFromEvent(payload);
  });
  flushPendingMetrics();
}

function openStream(jobId) {
  closeStream();
  setStreamBadge('idle');
  currentEventSource = new EventSource(`/api/jobs/${jobId}/events`);
  currentEventSource.onopen = () => {
    if (activeJobId !== jobId) {
      return;
    }
    setStreamBadge('connected');
  };
  currentEventSource.onerror = () => {
    if (activeJobId !== jobId) {
      return;
    }
    setStreamBadge('error');
  };
  currentEventSource.addEventListener('message', (event) => {
    try {
      const payload = JSON.parse(event.data);
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
  ['job_created', 'job_resumed', 'job_paused', 'dataset_loaded', 'model_started', 'row_generated', 'row_scored', 'model_completed', 'model_failed', 'job_completed', 'job_failed', 'job_cancelled'].forEach((name) => {
    currentEventSource.addEventListener(name, (event) => {
      if (activeJobId !== jobId) {
        return;
      }
      try {
        const payload = JSON.parse(event.data);
        addEvent(payload);
        updateLiveStatusFromEvent(payload);
        if (payload.event === 'job_paused') {
          setStreamBadge('paused');
          closeStream();
        } else if (TERMINAL_STATUSES.has(payload.event.replace('job_', ''))) {
          setStreamBadge('closed');
          closeStream();
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

function formatSessionLabel(job) {
  const timestamp = job?.started_at || job?.finished_at;
  const stamp = timestamp ? formatTimestamp(timestamp) : 'unknown time';
  return `${job.job_id} · ${job.status} · ${stamp}`;
}

async function loadSessions(preferredJobId = null) {
  const response = await fetch('/api/jobs');
  const sessions = await response.json();
  sessionSelect.innerHTML = '';
  if (!sessions.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'No sessions yet';
    sessionSelect.appendChild(option);
    sessionSelect.disabled = true;
    return [];
  }
  sessionSelect.disabled = false;
  sessions.forEach((job) => {
    const option = document.createElement('option');
    option.value = job.job_id;
    option.textContent = formatSessionLabel(job);
    sessionSelect.appendChild(option);
  });
  const selectedJobId = preferredJobId && sessions.some((job) => job.job_id === preferredJobId)
    ? preferredJobId
    : sessions[0].job_id;
  sessionSelect.value = selectedJobId;
  return sessions;
}

function resetDashboard() {
  closeStream();
  activeJobId = null;
  persistJobId(null);
  clearEventFeed();
  resetLiveStats();
  resetMetricsPlaceholder();
  setStreamBadge('idle');
  setOverview(null);
}

async function loadSession(jobId) {
  if (!jobId) {
    resetDashboard();
    return;
  }
  activeJobId = jobId;
  persistJobId(jobId);
  closeStream();
  clearEventFeed();
  resetLiveStats();
  setStreamBadge('idle');

  try {
    const [jobResponse, historyResponse] = await Promise.all([
      fetch(`/api/jobs/${jobId}`),
      fetch(`/api/jobs/${jobId}/history`),
    ]);
    if (!jobResponse.ok) {
      throw new Error(`Failed to load job ${jobId}`);
    }
    const job = await jobResponse.json();
    const events = historyResponse.ok ? await historyResponse.json() : [];
    if (activeJobId !== jobId) {
      return;
    }

    buildMetricsTable(job.request?.datasets || [], (job.request?.models || []).map(getNormalizedModelId));
    replayHistory(events);
    applySummaryMetrics(job);
    setOverview(job);

    if (!events.length) {
      addEvent({ event: 'client', message: `Loaded session ${jobId}`, data: job });
    }

    if (job.status === 'paused') {
      setStreamBadge('paused');
    } else if (TERMINAL_STATUSES.has(job.status)) {
      setStreamBadge('closed');
    } else {
      setJobBadge('running', 'live');
      openStream(jobId);
    }
  } catch (error) {
    console.warn('Unable to load session', error);
    setStreamBadge('error');
    setJobBadge('Unavailable', 'error');
    addEvent({ event: 'request_error', message: `Unable to load session ${jobId}`, data: { error: String(error) } });
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  setPendingAction('launch');
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

    activeJobId = data.job_id;
    persistJobId(data.job_id);
    clearEventFeed();
    resetLiveStats();
    buildMetricsTable(payload.datasets, payload.models);
    setOverview(data);
    setJobBadge('running', 'live');
    addEvent({ event: 'client', message: `Started job ${data.job_id}`, data });
    await loadSessions(data.job_id);
    openStream(data.job_id);
  } catch (error) {
    setJobBadge('Request failed', 'error');
    addEvent({ event: 'request_error', message: 'Network error while creating job.', data: { error: String(error) } });
  } finally {
    setPendingAction(null);
  }
});

continueButton.addEventListener('click', async () => {
  if (!activeJobId) {
    return;
  }
  setPendingAction('continue');
  try {
    const response = await fetch(`/api/jobs/${activeJobId}/resume`, { method: 'POST' });
    const data = await response.json();
    if (!response.ok) {
      setJobBadge('Request failed', 'error');
      addEvent({ event: 'request_error', message: data.detail || `Failed to continue ${activeJobId}.`, data });
      return;
    }
    setOverview(data);
    setStreamBadge('idle');
    setJobBadge('running', 'live');
    addEvent({ event: 'client', message: `Continuing job ${activeJobId}`, data });
    await loadSessions(activeJobId);
    openStream(activeJobId);
  } catch (error) {
    setJobBadge('Request failed', 'error');
    addEvent({ event: 'request_error', message: `Network error while continuing ${activeJobId}.`, data: { error: String(error) } });
  } finally {
    setPendingAction(null);
  }
});

pauseButton.addEventListener('click', async () => {
  if (!activeJobId) {
    return;
  }
  setPendingAction('pause');
  try {
    const response = await fetch(`/api/jobs/${activeJobId}/pause`, { method: 'POST' });
    const data = await response.json();
    if (!response.ok) {
      setJobBadge('Request failed', 'error');
      addEvent({ event: 'request_error', message: data.detail || `Failed to pause ${activeJobId}.`, data });
      return;
    }
    setOverview(data);
    setStreamBadge('paused');
    await loadSessions(activeJobId);
  } catch (error) {
    setJobBadge('Request failed', 'error');
    addEvent({ event: 'request_error', message: `Network error while pausing ${activeJobId}.`, data: { error: String(error) } });
  } finally {
    setPendingAction(null);
  }
});

sessionSelect.addEventListener('change', async () => {
  await loadSession(sessionSelect.value);
});

clearLog.addEventListener('click', () => {
  clearEventFeed();
});

async function init() {
  await loadConfig();
  const preferredJobId = readPersistedJobId();
  const sessions = await loadSessions(preferredJobId);
  const initialJobId = preferredJobId && sessions.some((job) => job.job_id === preferredJobId)
    ? preferredJobId
    : sessions[0]?.job_id;
  if (initialJobId) {
    await loadSession(initialJobId);
  } else {
    resetDashboard();
  }
}

init();
