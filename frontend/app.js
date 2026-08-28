/**
 * app.js — Semantic Code Intelligence frontend
 *
 * Responsibilities:
 *  - Tab navigation (Search / Index Repo)
 *  - Health-check polling with status indicator
 *  - Search form → GET /api/search → render result cards
 *    • repo dropdown populated from GET /api/repos
 *    • repo_name filter wired into search params
 *  - Index form → POST /api/index → show progress + summary
 *  - Repo management card → list repos, delete individual repos
 */

"use strict";

const API_BASE          = "";
const HEALTH_INTERVAL_MS = 30_000;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = (sel, ctx = document) => ctx.querySelector(sel);

const statusDot    = $("#status-dot");
const statusText   = $("#status-text");
const tabBtns      = [...document.querySelectorAll(".tab-btn")];
const tabPanels    = [...document.querySelectorAll(".tab-panel")];

// Search panel
const searchInput  = $("#search-input");
const searchBtn    = $("#search-btn");
const filterLang   = $("#filter-language");
const filterRepo   = $("#filter-repo");
const filterTopK   = $("#filter-topk");
const searchAlert  = $("#search-alert");
const searchStats  = $("#search-stats");
const resultsList  = $("#results-list");
const statQuery    = $("#stat-query");
const statCount    = $("#stat-count");
const statTotal    = $("#stat-total");
const statTopScore = $("#stat-top-score");

// Index panel
const srcLocalBtn      = $("#src-local-btn");
const srcUrlBtn        = $("#src-url-btn");
const srcLocalDiv      = $("#src-local");
const srcUrlDiv        = $("#src-url");
const repoPath         = $("#repo-path");
const repoUrl          = $("#repo-url");
const indexLang        = $("#index-language");
const repoName         = $("#repo-name");
const indexBtn         = $("#index-btn");
const healthBtn        = $("#health-btn");
const indexProgress    = $("#index-progress");
const progressFill     = $("#progress-fill");
const indexAlert       = $("#index-alert");
const indexStats       = $("#index-stats");
const istatRepo        = $("#istat-repo");
const istatChunks      = $("#istat-chunks");
const istatFiles       = $("#istat-files");
const istatSkipped     = $("#istat-skipped");
const istatDuration    = $("#istat-duration");

// Repo management
const refreshReposBtn    = $("#refresh-repos-btn");
const deleteCloneToggle  = $("#delete-clone-toggle");
const repoListEl         = $("#repo-list");
const manageAlert        = $("#manage-alert");


// ── Utilities ─────────────────────────────────────────────────────────────────

function showAlert(container, message, type = "info") {
  container.innerHTML = `<div class="alert alert-${type}" role="alert"><span>${escapeHtml(message)}</span></div>`;
}
function clearAlert(container) { container.innerHTML = ""; }

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function escapeCode(str) {
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function truncate(str, maxLen = 60) {
  if (!str) return "";
  return str.length > maxLen ? str.slice(0, maxLen) + "…" : str;
}
function pct(val) { return (val * 100).toFixed(1) + "%"; }

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail ?? detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}


// ── Tab navigation ─────────────────────────────────────────────────────────────

function activateTab(tabName) {
  tabBtns.forEach(btn => {
    const active = btn.dataset.tab === tabName;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  tabPanels.forEach(panel => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
  // Load repo list whenever Index tab becomes visible
  if (tabName === "index") loadRepos();
}

tabBtns.forEach(btn => btn.addEventListener("click", () => activateTab(btn.dataset.tab)));


// ── Health check ───────────────────────────────────────────────────────────────

async function checkHealth(quiet = false) {
  try {
    const data = await apiFetch("/api/health");
    statusDot.className   = "status-dot ok";
    statusText.textContent = `API ok · ${data.vector_store_size ?? 0} chunks`;
    if (!quiet) {
      showAlert(indexAlert,
        `API healthy · model: ${data.embedding_model} · chunks: ${data.vector_store_size} · uptime: ${data.uptime_seconds}s`,
        "success");
    }
  } catch (err) {
    statusDot.className   = "status-dot err";
    statusText.textContent = "API unreachable";
    if (!quiet) showAlert(indexAlert, `Health check failed: ${err.message}`, "error");
  }
}

healthBtn.addEventListener("click", () => checkHealth(false));
checkHealth(true);
setInterval(() => checkHealth(true), HEALTH_INTERVAL_MS);


// ── Repo dropdown (search panel) ───────────────────────────────────────────────

/**
 * Fetch GET /api/repos and populate the filter-repo <select>.
 * Preserves the currently selected value if it still exists.
 */
async function refreshRepoDropdown() {
  try {
    const data = await apiFetch("/api/repos");
    const current = filterRepo.value;

    // Keep "All repositories" as first option, then add one per repo.
    filterRepo.innerHTML = '<option value="">All repositories</option>';
    (data.repos || []).forEach(r => {
      const opt = document.createElement("option");
      opt.value       = r.repo_name;
      opt.textContent = `${r.repo_name} (${r.chunk_count.toLocaleString()} chunks)`;
      filterRepo.appendChild(opt);
    });

    // Restore selection if it still exists.
    if (current && [...filterRepo.options].some(o => o.value === current)) {
      filterRepo.value = current;
    }
  } catch (_) {
    // Silently ignore — dropdown stays as "All repositories".
  }
}

// Populate on first load.
refreshRepoDropdown();


// ── Search ─────────────────────────────────────────────────────────────────────

async function runSearch() {
  const query = searchInput.value.trim();
  if (!query) { showAlert(searchAlert, "Please enter a search query.", "warning"); return; }

  clearAlert(searchAlert);
  searchBtn.disabled    = true;
  searchBtn.innerHTML   = '<span class="spinner" aria-hidden="true"></span> Searching…';
  resultsList.innerHTML = "";
  searchStats.hidden    = true;

  const params = new URLSearchParams({ q: query });
  const topK = parseInt(filterTopK.value, 10);
  if (!isNaN(topK) && topK > 0) params.set("top_k", topK);
  if (filterLang.value)  params.set("language",  filterLang.value);
  if (filterRepo.value)  params.set("repo_name", filterRepo.value);

  try {
    const data = await apiFetch(`/api/search?${params}`);
    renderResults(data);
  } catch (err) {
    showAlert(searchAlert, err.message, "error");
    resultsList.innerHTML = `<div class="empty-state"><div class="icon" aria-hidden="true">⚠️</div><p>Search failed. Check the alert above.</p></div>`;
  } finally {
    searchBtn.disabled    = false;
    searchBtn.textContent = "Search";
  }
}

function renderResults(data) {
  statQuery.textContent    = truncate(data.query, 50);
  statCount.textContent    = data.results.length;
  statTotal.textContent    = data.total_indexed.toLocaleString();
  statTopScore.textContent = data.results.length ? pct(data.results[0].score) : "—";
  searchStats.hidden       = false;

  if (!data.results.length) {
    resultsList.innerHTML = `
      <div class="empty-state">
        <div class="icon" aria-hidden="true">🔍</div>
        <p>No results found for <strong>${escapeHtml(data.query)}</strong>.
           Try a broader query or choose a different repository.</p>
      </div>`;
    return;
  }
  resultsList.innerHTML = data.results.map((r, i) => buildResultCard(r, i + 1)).join("");
}

function buildResultCard(r, rank) {
  const symbol   = r.symbol_name ? escapeHtml(r.symbol_name) : "(anonymous)";
  const lang     = r.language   ?? "unknown";
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
    <span class="badge badge-lang"  title="Language">${escapeHtml(lang)}</span>
    <span class="badge badge-type"  title="Chunk type">${escapeHtml(type)}</span>
    <span class="badge badge-score" title="Fused relevance score">${pct(r.score)}</span>
    <span class="result-filepath"   title="${filepath}">${filepath}</span>
  </div>
  <div class="result-scores" aria-label="Score breakdown">
    <span class="score-chip"><span>sem</span><span>${pct(r.semantic_score  ?? 0)}</span></span>
    <span class="score-chip"><span>bm25</span><span>${pct(r.bm25_score     ?? 0)}</span></span>
    <span class="score-chip"><span>sym</span><span>${pct(r.symbol_score    ?? 0)}</span></span>
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

window.copyCode = async function copyCode(btn) {
  const code = btn.closest(".result-card")?.querySelector("pre.result-code")?.textContent ?? "";
  try {
    await navigator.clipboard.writeText(code);
    const orig = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = orig; }, 1500);
  } catch (_) { btn.textContent = "Failed"; }
};

searchBtn.addEventListener("click", runSearch);
searchInput.addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });


// ── Index form — source toggle ────────────────────────────────────────────────

let sourceMode = "local";

function setSourceMode(mode) {
  sourceMode = mode;
  const isLocal = mode === "local";
  srcLocalDiv.hidden        = !isLocal;
  srcUrlDiv.hidden          =  isLocal;
  srcLocalBtn.className     = isLocal ? "btn btn-primary"   : "btn btn-secondary";
  srcUrlBtn.className       = isLocal ? "btn btn-secondary" : "btn btn-primary";
  srcLocalBtn.setAttribute("aria-pressed", String(isLocal));
  srcUrlBtn.setAttribute("aria-pressed",   String(!isLocal));
}

srcLocalBtn.addEventListener("click", () => setSourceMode("local"));
srcUrlBtn.addEventListener("click",   () => setSourceMode("url"));


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
  if (indexLang.value)        body.language  = indexLang.value;
  if (repoName.value.trim())  body.repo_name = repoName.value.trim();

  indexBtn.disabled      = true;
  indexBtn.innerHTML     = '<span class="spinner" aria-hidden="true"></span> Indexing…';
  indexProgress.hidden   = false;
  progressFill.className = "progress-fill indeterminate";

  try {
    const data = await apiFetch("/api/index", { method: "POST", body: JSON.stringify(body) });

    progressFill.className  = "progress-fill";
    progressFill.style.width = "100%";

    istatRepo.textContent     = data.repo_name;
    istatChunks.textContent   = data.chunks_indexed.toLocaleString();
    istatFiles.textContent    = data.files_processed.toLocaleString();
    istatSkipped.textContent  = data.skipped_files.toLocaleString();
    istatDuration.textContent = `${data.duration_seconds}s`;
    indexStats.hidden = false;

    showAlert(indexAlert,
      `Indexed ${data.chunks_indexed.toLocaleString()} chunks from "${data.repo_name}" in ${data.duration_seconds}s.`,
      "success");

    // Refresh both header and the repo list + search dropdown.
    checkHealth(true);
    loadRepos();
    refreshRepoDropdown();
  } catch (err) {
    progressFill.className   = "progress-fill";
    progressFill.style.width = "0%";
    showAlert(indexAlert, `Indexing failed: ${err.message}`, "error");
  } finally {
    indexBtn.disabled     = false;
    indexBtn.textContent  = "Index repository";
    setTimeout(() => { indexProgress.hidden = true; }, 1200);
  }
}

indexBtn.addEventListener("click", runIndex);


// ── Repo management ───────────────────────────────────────────────────────────

async function loadRepos() {
  repoListEl.innerHTML = `<p style="color:var(--text-muted);font-size:0.85rem;">Loading…</p>`;
  clearAlert(manageAlert);

  try {
    const data = await apiFetch("/api/repos");

    if (!data.repos || data.repos.length === 0) {
      repoListEl.innerHTML = `
        <div class="empty-state" style="padding:1.5rem 0;">
          <p>No repositories indexed yet.</p>
        </div>`;
      return;
    }

    repoListEl.innerHTML = data.repos.map(r => buildRepoRow(r)).join("");
  } catch (err) {
    repoListEl.innerHTML = "";
    showAlert(manageAlert, `Failed to load repos: ${err.message}`, "error");
  }
}

function buildRepoRow(r) {
  const cloneInfo = r.has_local_clone
    ? `<span class="badge badge-type" title="Clone on disk">📁 ${r.clone_size_mb} MB</span>`
    : `<span class="badge" style="background:rgba(139,144,176,0.15);color:var(--text-muted);">no clone</span>`;

  return `
<div class="repo-row" id="repo-row-${CSS.escape(r.repo_name)}"
     style="display:flex;align-items:center;justify-content:space-between;
            gap:0.75rem;padding:0.65rem 0;border-bottom:1px solid var(--border);flex-wrap:wrap;">
  <div style="display:flex;align-items:center;gap:0.6rem;min-width:0;">
    <span style="font-family:var(--font-mono);font-size:0.88rem;font-weight:600;
                 color:var(--accent);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
      ${escapeHtml(r.repo_name)}
    </span>
    <span class="badge badge-score">${r.chunk_count.toLocaleString()} chunks</span>
    ${cloneInfo}
  </div>
  <button class="btn btn-danger"
          style="font-size:0.78rem;padding:0.3rem 0.75rem;flex-shrink:0;"
          aria-label="Remove ${escapeHtml(r.repo_name)}"
          onclick="confirmDeleteRepo('${escapeHtml(r.repo_name)}')">
    Remove
  </button>
</div>`;
}

window.confirmDeleteRepo = async function confirmDeleteRepo(repoName) {
  const deleteClone = deleteCloneToggle.checked;
  const cloneNote   = deleteClone ? " and delete its source files from disk" : "";
  const confirmed   = confirm(
    `Remove "${repoName}" from the index${cloneNote}?\n\nThis cannot be undone.`
  );
  if (!confirmed) return;

  clearAlert(manageAlert);

  // Visually mark the row as being removed.
  const row = document.getElementById(`repo-row-${CSS.escape(repoName)}`);
  if (row) {
    row.style.opacity = "0.4";
    row.style.pointerEvents = "none";
  }

  try {
    const params = new URLSearchParams({ delete_clone: deleteClone });
    const data   = await apiFetch(`/api/repos/${encodeURIComponent(repoName)}?${params}`, {
      method: "DELETE",
    });

    showAlert(manageAlert, data.message, "success");

    // Refresh everything that depends on the repo list.
    loadRepos();
    refreshRepoDropdown();
    checkHealth(true);
  } catch (err) {
    if (row) { row.style.opacity = ""; row.style.pointerEvents = ""; }
    showAlert(manageAlert, `Failed to remove "${repoName}": ${err.message}`, "error");
  }
};

refreshReposBtn.addEventListener("click", loadRepos);
