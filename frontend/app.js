/**
 * app.js — Semantic Code Intelligence frontend
 *
 * Responsibilities:
 *  - Tab navigation (Search / Index Repo)
 *  - Health-check polling with status indicator
 *  - Search form → GET /api/search → render result cards
 *  - Index form → POST /api/index → show progress + summary
 *
 * No build step, no framework — plain ES2021 modules loaded directly.
 * The API base URL is auto-detected as the same origin that serves this page.
 */

"use strict";

// ── Config ──────────────────────────────────────────────────────────────────
const API_BASE = "";   // same origin; change to "http://localhost:8000" for dev
const HEALTH_INTERVAL_MS = 30_000;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

const statusDot    = $("#status-dot");
const statusText   = $("#status-text");

const tabBtns      = $$(".tab-btn");
const tabPanels    = $$(".tab-panel");

// Search panel
const searchInput  = $("#search-input");
const searchBtn    = $("#search-btn");
const filterLang   = $("#filter-language");
const filterTopK   = $("#filter-topk");
const searchAlert  = $("#search-alert");
const searchStats  = $("#search-stats");
const resultsList  = $("#results-list");
const statQuery    = $("#stat-query");
const statCount    = $("#stat-count");
const statTotal    = $("#stat-total");
const statTopScore = $("#stat-top-score");

// Index panel
const srcLocalBtn  = $("#src-local-btn");
const srcUrlBtn    = $("#src-url-btn");
const srcLocalDiv  = $("#src-local");
const srcUrlDiv    = $("#src-url");
const repoPath     = $("#repo-path");
const repoUrl      = $("#repo-url");
const indexLang    = $("#index-language");
const repoName     = $("#repo-name");
const indexBtn     = $("#index-btn");
const healthBtn    = $("#health-btn");
const indexProgress = $("#index-progress");
const progressFill = $("#progress-fill");
const indexAlert   = $("#index-alert");
const indexStats   = $("#index-stats");
const istatRepo    = $("#istat-repo");
const istatChunks  = $("#istat-chunks");
const istatFiles   = $("#istat-files");
const istatSkipped = $("#istat-skipped");
const istatDuration = $("#istat-duration");


// ── Utilities ─────────────────────────────────────────────────────────────────

/** Show an alert message inside `container`. */
function showAlert(container, message, type = "info") {
  container.innerHTML = `
    <div class="alert alert-${type}" role="alert">
      <span>${escapeHtml(message)}</span>
    </div>`;
}

function clearAlert(container) {
  container.innerHTML = "";
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeCode(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/** Truncate a string to maxLen chars with ellipsis. */
function truncate(str, maxLen = 60) {
  if (!str) return "";
  return str.length > maxLen ? str.slice(0, maxLen) + "…" : str;
}

/** Format a float as a percentage string, e.g. 0.873 → "87.3%" */
function pct(val) {
  return (val * 100).toFixed(1) + "%";
}

/**
 * Lightweight fetch wrapper with JSON body/response handling.
 * Throws an Error with the server's `detail` message on non-2xx responses.
 */
async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail ?? detail;
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }
  return res.json();
}


// ── Tab navigation ────────────────────────────────────────────────────────────

function activateTab(tabName) {
  tabBtns.forEach(btn => {
    const active = btn.dataset.tab === tabName;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  tabPanels.forEach(panel => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
}

tabBtns.forEach(btn => {
  btn.addEventListener("click", () => activateTab(btn.dataset.tab));
});


// ── Health check ──────────────────────────────────────────────────────────────

async function checkHealth(quiet = false) {
  try {
    const data = await apiFetch("/api/health");
    statusDot.className = "status-dot ok";
    statusText.textContent =
      `API ok · ${data.vector_store_size ?? 0} chunks`;

    if (!quiet) {
      showAlert(
        indexAlert,
        `API healthy · model: ${data.embedding_model} · ` +
        `chunks: ${data.vector_store_size} · uptime: ${data.uptime_seconds}s`,
        "success",
      );
    }
  } catch (err) {
    statusDot.className = "status-dot err";
    statusText.textContent = "API unreachable";
    if (!quiet) showAlert(indexAlert, `Health check failed: ${err.message}`, "error");
  }
}

healthBtn.addEventListener("click", () => checkHealth(false));

// Poll health silently after the initial check.
checkHealth(true);
setInterval(() => checkHealth(true), HEALTH_INTERVAL_MS);


// ── Search ────────────────────────────────────────────────────────────────────

async function runSearch() {
  const query = searchInput.value.trim();
  if (!query) {
    showAlert(searchAlert, "Please enter a search query.", "warning");
    return;
  }

  clearAlert(searchAlert);
  searchBtn.disabled = true;
  searchBtn.innerHTML = '<span class="spinner" aria-hidden="true"></span> Searching…';
  resultsList.innerHTML = "";
  searchStats.hidden = true;

  const params = new URLSearchParams({ q: query });
  const topK = parseInt(filterTopK.value, 10);
  if (!isNaN(topK) && topK > 0) params.set("top_k", topK);
  const lang = filterLang.value;
  if (lang) params.set("language", lang);

  try {
    const data = await apiFetch(`/api/search?${params}`);
    renderResults(data);
  } catch (err) {
    showAlert(searchAlert, err.message, "error");
    resultsList.innerHTML = `
      <div class="empty-state">
        <div class="icon" aria-hidden="true">⚠️</div>
        <p>Search failed. Check the alert above.</p>
      </div>`;
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = "Search";
  }
}

function renderResults(data) {
  // Update stats bar
  statQuery.textContent = truncate(data.query, 50);
  statCount.textContent = data.results.length;
  statTotal.textContent = data.total_indexed.toLocaleString();
  statTopScore.textContent = data.results.length
    ? pct(data.results[0].score)
    : "—";
  searchStats.hidden = false;

  if (!data.results.length) {
    resultsList.innerHTML = `
      <div class="empty-state">
        <div class="icon" aria-hidden="true">🔍</div>
        <p>No results found for <strong>${escapeHtml(data.query)}</strong>.
           Try a broader query or index more repositories.</p>
      </div>`;
    return;
  }

  resultsList.innerHTML = data.results
    .map((r, i) => buildResultCard(r, i + 1))
    .join("");
}

function buildResultCard(r, rank) {
  const symbol   = r.symbol_name ? escapeHtml(r.symbol_name) : "(anonymous)";
  const lang     = r.language ?? "unknown";
  const type     = r.chunk_type ?? "block";
  const filepath = escapeHtml(r.file_path ?? "");
  const code     = escapeCode(r.code ?? "");
  const lines    = `lines ${r.start_line}–${r.end_line}`;
  const repo     = r.repo_name ? escapeHtml(r.repo_name) : "";

  return `
<article class="result-card" aria-label="Result ${rank}: ${symbol}">
  <div class="result-header">
    <span class="result-rank" aria-label="Rank">#${rank}</span>
    <span class="result-symbol">${symbol}</span>
    <span class="badge badge-lang" title="Language">${escapeHtml(lang)}</span>
    <span class="badge badge-type" title="Chunk type">${escapeHtml(type)}</span>
    <span class="badge badge-score" title="Fused relevance score">${pct(r.score)}</span>
    <span class="result-filepath" title="${filepath}">${filepath}</span>
  </div>
  <div class="result-scores" aria-label="Score breakdown">
    <span class="score-chip">
      <span>sem</span><span>${pct(r.semantic_score ?? 0)}</span>
    </span>
    <span class="score-chip">
      <span>bm25</span><span>${pct(r.bm25_score ?? 0)}</span>
    </span>
    <span class="score-chip">
      <span>sym</span><span>${pct(r.symbol_score ?? 0)}</span>
    </span>
  </div>
  <pre class="result-code" tabindex="0" aria-label="Source code"><code>${code}</code></pre>
  <div class="result-footer">
    <span class="result-lines">${lines}</span>
    ${repo ? `<span class="badge badge-lang" title="Repository">${repo}</span>` : ""}
    <button class="btn btn-secondary" style="font-size:0.75rem;padding:0.25rem 0.6rem"
            onclick="copyCode(this)" aria-label="Copy code to clipboard">Copy</button>
  </div>
</article>`;
}

/** Copy the code from the nearest result-card to clipboard. */
window.copyCode = async function copyCode(btn) {
  const card = btn.closest(".result-card");
  const code = card?.querySelector("pre.result-code")?.textContent ?? "";
  try {
    await navigator.clipboard.writeText(code);
    const orig = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = orig; }, 1500);
  } catch (_) {
    btn.textContent = "Failed";
  }
};

searchBtn.addEventListener("click", runSearch);
searchInput.addEventListener("keydown", e => {
  if (e.key === "Enter") runSearch();
});


// ── Index form — source toggle ────────────────────────────────────────────────

let sourceMode = "local";

function setSourceMode(mode) {
  sourceMode = mode;
  if (mode === "local") {
    srcLocalDiv.hidden = false;
    srcUrlDiv.hidden   = true;
    srcLocalBtn.className = "btn btn-primary";
    srcLocalBtn.setAttribute("aria-pressed", "true");
    srcUrlBtn.className   = "btn btn-secondary";
    srcUrlBtn.setAttribute("aria-pressed", "false");
  } else {
    srcLocalDiv.hidden = true;
    srcUrlDiv.hidden   = false;
    srcLocalBtn.className = "btn btn-secondary";
    srcLocalBtn.setAttribute("aria-pressed", "false");
    srcUrlBtn.className   = "btn btn-primary";
    srcUrlBtn.setAttribute("aria-pressed", "true");
  }
}

srcLocalBtn.addEventListener("click", () => setSourceMode("local"));
srcUrlBtn.addEventListener("click", () => setSourceMode("url"));


// ── Index repo ────────────────────────────────────────────────────────────────

async function runIndex() {
  clearAlert(indexAlert);
  indexStats.hidden = true;

  const body = {};
  if (sourceMode === "local") {
    const p = repoPath.value.trim();
    if (!p) { showAlert(indexAlert, "Enter a local directory path.", "warning"); return; }
    body.path = p;
  } else {
    const u = repoUrl.value.trim();
    if (!u) { showAlert(indexAlert, "Enter a git repository URL.", "warning"); return; }
    body.repo_url = u;
  }
  if (indexLang.value) body.language = indexLang.value;
  if (repoName.value.trim()) body.repo_name = repoName.value.trim();

  indexBtn.disabled = true;
  indexBtn.innerHTML = '<span class="spinner" aria-hidden="true"></span> Indexing…';
  indexProgress.hidden = false;
  progressFill.className = "progress-fill indeterminate";

  try {
    const data = await apiFetch("/api/index", {
      method: "POST",
      body: JSON.stringify(body),
    });

    // Stop animation and fill bar to 100%
    progressFill.className = "progress-fill";
    progressFill.style.width = "100%";

    istatRepo.textContent     = data.repo_name;
    istatChunks.textContent   = data.chunks_indexed.toLocaleString();
    istatFiles.textContent    = data.files_processed.toLocaleString();
    istatSkipped.textContent  = data.skipped_files.toLocaleString();
    istatDuration.textContent = `${data.duration_seconds}s`;
    indexStats.hidden = false;

    showAlert(
      indexAlert,
      `Indexed ${data.chunks_indexed.toLocaleString()} chunks from "${data.repo_name}" in ${data.duration_seconds}s.`,
      "success",
    );

    // Refresh health stats in the header
    checkHealth(true);
  } catch (err) {
    progressFill.className = "progress-fill";
    progressFill.style.width = "0%";
    showAlert(indexAlert, `Indexing failed: ${err.message}`, "error");
  } finally {
    indexBtn.disabled = false;
    indexBtn.textContent = "Index repository";
    setTimeout(() => { indexProgress.hidden = true; }, 1200);
  }
}

indexBtn.addEventListener("click", runIndex);
