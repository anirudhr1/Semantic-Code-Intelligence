/**
 * app.js — Semantic Code Intelligence frontend
 *
 * Features:
 *  - Theme switcher (dark / light) with highlight.js sync & localStorage
 *  - First-time welcome banner with dismiss state
 *  - Example queries & recent search history pills
 *  - Tab navigation (Search / Index Repo)
 *  - Health-check polling with status indicator
 *  - Search form with multi-repo filter support & loading skeletons
 *  - Rich result cards: syntax highlighting, line numbers gutter,
 *    visual score bars, match tiers, and animated copy button
 *  - Incremental indexing summary with unchanged file counts
 *  - Toast notification system
 */

"use strict";

const API_BASE          = "";
const HEALTH_INTERVAL_MS = 30_000;
const THEME_KEY         = "sci-theme";
const WELCOME_KEY       = "sci-welcome-dismissed";
const RECENT_KEY        = "sci-recent-searches";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = (sel, ctx = document) => ctx.querySelector(sel);

// Theme & Welcome
const themeToggle      = $("#theme-toggle");
const themeIcon        = $("#theme-icon");
const welcomeBanner    = $("#welcome-banner");
const welcomeDismiss   = $("#welcome-dismiss");

// Status & Tabs
const statusDot        = $("#status-dot");
const statusText       = $("#status-text");
const tabBtns          = [...document.querySelectorAll(".tab-btn")];
const tabPanels        = [...document.querySelectorAll(".tab-panel")];

// Search panel
const searchInput      = $("#search-input");
const searchBtn        = $("#search-btn");
const filterLang       = $("#filter-language");
const filterRepo       = $("#filter-repo");
const filterTopK       = $("#filter-topk");
const searchAlert      = $("#search-alert");
const searchStats      = $("#search-stats");
const resultsList      = $("#results-list");
const statQuery        = $("#stat-query");
const statCount        = $("#stat-count");
const statTotal        = $("#stat-total");
const statTopScore     = $("#stat-top-score");
const recentSearches   = $("#recent-searches");
const recentPills      = $("#recent-pills");
const recentClear      = $("#recent-clear");

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
const progressLabel    = $("#progress-label");
const indexAlert       = $("#index-alert");
const indexStats       = $("#index-stats");
const istatRepo        = $("#istat-repo");
const istatChunks      = $("#istat-chunks");
const istatFiles       = $("#istat-files");
const istatSkipped     = $("#istat-skipped");
const istatUnchanged   = $("#istat-unchanged");
const istatDuration    = $("#istat-duration");

// Repo management
const refreshReposBtn   = $("#refresh-repos-btn");
const deleteCloneToggle = $("#delete-clone-toggle");
const repoListEl        = $("#repo-list");
const manageAlert       = $("#manage-alert");
const toastContainer    = $("#toast-container");


// ── Theme Switcher ────────────────────────────────────────────────────────────

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem(THEME_KEY, theme);
  const isDark = theme === "dark";
  if (themeIcon) themeIcon.textContent = isDark ? "🌙" : "☀️";

  const darkHljs = $("#hljs-dark-theme");
  const lightHljs = $("#hljs-light-theme");
  if (darkHljs) darkHljs.disabled = !isDark;
  if (lightHljs) lightHljs.disabled = isDark;
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY) || "dark";
  setTheme(saved);
}

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    setTheme(current === "dark" ? "light" : "dark");
  });
}


// ── Welcome Banner ────────────────────────────────────────────────────────────

function initWelcomeBanner() {
  if (welcomeBanner && !localStorage.getItem(WELCOME_KEY)) {
    welcomeBanner.hidden = false;
  }
}

if (welcomeDismiss) {
  welcomeDismiss.addEventListener("click", () => {
    if (welcomeBanner) welcomeBanner.hidden = true;
    localStorage.setItem(WELCOME_KEY, "true");
  });
}


// ── Toast Notifications ───────────────────────────────────────────────────────

function showToast(message, type = "info", durationMs = 3000) {
  if (!toastContainer) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.setAttribute("role", "status");
  toast.innerHTML = `<span>${escapeHtml(message)}</span>`;
  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("toast-fade-out");
    setTimeout(() => toast.remove(), 300);
  }, durationMs);
}


// ── Utilities ─────────────────────────────────────────────────────────────────

function showAlert(container, message, type = "info") {
  if (!container) return;
  container.innerHTML = `<div class="alert alert-${type}" role="alert"><span>${escapeHtml(message)}</span></div>`;
}

function clearAlert(container) {
  if (container) container.innerHTML = "";
}

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

function pct(val) {
  return (Math.max(0, val) * 100).toFixed(1) + "%";
}

function getScoreTier(score) {
  if (score >= 0.70) return { tier: "excellent", label: "Excellent match" };
  if (score >= 0.45) return { tier: "good",      label: "Good match" };
  return { tier: "partial", label: "Partial match" };
}

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


// ── Tab Navigation ────────────────────────────────────────────────────────────

function activateTab(tabName) {
  tabBtns.forEach(btn => {
    const active = btn.dataset.tab === tabName;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", String(active));
  });
  tabPanels.forEach(panel => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });
  if (tabName === "index") loadRepos();
}

tabBtns.forEach(btn => btn.addEventListener("click", () => activateTab(btn.dataset.tab)));


// ── Health Check ──────────────────────────────────────────────────────────────

async function checkHealth(quiet = false) {
  try {
    const data = await apiFetch("/api/health");
    statusDot.className    = "status-dot ok";
    statusText.textContent = `API ok · ${data.vector_store_size ?? 0} chunks`;
    if (!quiet) {
      showAlert(indexAlert,
        `API healthy · model: ${data.embedding_model} · chunks: ${data.vector_store_size} · uptime: ${data.uptime_seconds}s`,
        "success");
      showToast("Backend service is healthy", "success");
    }
  } catch (err) {
    statusDot.className    = "status-dot err";
    statusText.textContent = "API unreachable";
    if (!quiet) {
      showAlert(indexAlert, `Health check failed: ${err.message}`, "error");
      showToast(`Health check failed: ${err.message}`, "error");
    }
  }
}

healthBtn.addEventListener("click", () => checkHealth(false));
checkHealth(true);
setInterval(() => checkHealth(true), HEALTH_INTERVAL_MS);


// ── Repo Dropdown (search panel) ──────────────────────────────────────────────

async function refreshRepoDropdown() {
  try {
    const data = await apiFetch("/api/repos");
    const current = filterRepo.value;

    filterRepo.innerHTML = '<option value="">All repositories</option>';
    (data.repos || []).forEach(r => {
      const opt = document.createElement("option");
      opt.value       = r.repo_name;
      opt.textContent = `${r.repo_name} (${r.chunk_count.toLocaleString()} chunks)`;
      filterRepo.appendChild(opt);
    });

    if (current && [...filterRepo.options].some(o => o.value === current)) {
      filterRepo.value = current;
    }
  } catch (_) {
    // Silently ignore — dropdown stays as "All repositories".
  }
}

refreshRepoDropdown();


// ── Search History & Example Queries ──────────────────────────────────────────

function getRecentSearches() {
  try { return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]"); } catch { return []; }
}

function addRecentSearch(query) {
  const trimmed = query.trim();
  if (!trimmed) return;
  let list = getRecentSearches().filter(q => q.toLowerCase() !== trimmed.toLowerCase());
  list.unshift(trimmed);
  if (list.length > 6) list = list.slice(0, 6);
  localStorage.setItem(RECENT_KEY, JSON.stringify(list));
  renderRecentSearches();
}

function renderRecentSearches() {
  if (!recentSearches || !recentPills) return;
  const list = getRecentSearches();
  if (list.length === 0) {
    recentSearches.hidden = true;
    return;
  }
  recentSearches.hidden = false;
  recentPills.innerHTML = list
    .map(q => `<button class="recent-pill" data-query="${escapeHtml(q)}">${escapeHtml(q)}</button>`)
    .join("");

  recentPills.querySelectorAll(".recent-pill").forEach(btn => {
    btn.addEventListener("click", () => {
      searchInput.value = btn.dataset.query;
      runSearch();
    });
  });
}

if (recentClear) {
  recentClear.addEventListener("click", () => {
    localStorage.removeItem(RECENT_KEY);
    renderRecentSearches();
  });
}

document.querySelectorAll(".example-pill").forEach(pill => {
  pill.addEventListener("click", () => {
    const q = pill.dataset.query;
    if (q) {
      searchInput.value = q;
      activateTab("search");
      runSearch();
    }
  });
});


// ── Loading Skeletons ─────────────────────────────────────────────────────────

function renderSkeletons(count = 3) {
  resultsList.innerHTML = Array.from({ length: count }, () => `
    <div class="result-card skeleton-card" aria-hidden="true">
      <div class="skeleton-line medium"></div>
      <div class="skeleton-line short"></div>
      <div class="skeleton-code"></div>
    </div>
  `).join("");
}


// ── Search Logic ──────────────────────────────────────────────────────────────

async function runSearch() {
  const query = searchInput.value.trim();
  if (!query) {
    showAlert(searchAlert, "Please enter a search query.", "warning");
    return;
  }

  clearAlert(searchAlert);
  searchBtn.disabled    = true;
  searchBtn.innerHTML   = '<span class="spinner" aria-hidden="true"></span> Searching…';
  searchStats.hidden    = true;
  renderSkeletons(3);

  const params = new URLSearchParams({ q: query });
  const topK = parseInt(filterTopK.value, 10);
  if (!isNaN(topK) && topK > 0) params.set("top_k", topK);
  if (filterLang.value)  params.set("language",  filterLang.value);
  if (filterRepo.value)  params.set("repo_name", filterRepo.value);

  try {
    const data = await apiFetch(`/api/search?${params}`);
    addRecentSearch(query);
    renderResults(data);
  } catch (err) {
    showAlert(searchAlert, err.message, "error");
    resultsList.innerHTML = `
      <div class="empty-state">
        <div class="icon" aria-hidden="true">⚠️</div>
        <p>Search failed. Check the alert above.</p>
      </div>`;
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

  // Apply syntax highlighting to rendered code blocks
  if (window.hljs) {
    resultsList.querySelectorAll("pre.result-code code").forEach(el => {
      try {
        window.hljs.highlightElement(el);
      } catch (_) {
        // Fallback to unhighlighted code if language parser fails
      }
    });
  }
}

function buildResultCard(r, rank) {
  const symbol   = r.symbol_name ? escapeHtml(r.symbol_name) : "(anonymous)";
  const lang     = r.language   ?? "unknown";
  const type     = r.chunk_type ?? "block";
  const filepath = escapeHtml(r.file_path ?? "");
  const code     = escapeCode(r.code ?? "");
  const lines    = `lines ${r.start_line}–${r.end_line}`;
  const repo     = r.repo_name ? escapeHtml(r.repo_name) : "";

  // Scoring
  const fusedPct = Math.round((r.score ?? 0) * 100);
  const semPct   = Math.round((r.semantic_score ?? 0) * 100);
  const bm25Pct  = Math.round((r.bm25_score ?? 0) * 100);
  const symPct   = Math.round((r.symbol_score ?? 0) * 100);

  const { tier, label } = getScoreTier(r.score ?? 0);
  const semTier = getScoreTier(r.semantic_score ?? 0).tier;
  const bm25Tier = getScoreTier(r.bm25_score ?? 0).tier;
  const symTier = getScoreTier(r.symbol_score ?? 0).tier;

  // Generate line numbers for gutter
  const lineCount = (r.code ?? "").split("\n").length;
  const lineNumbers = Array.from({ length: lineCount }, (_, idx) => r.start_line + idx).join("\n");

  // Highlight.js class mapping
  const hljsLang = (lang && lang !== "unknown") ? `language-${escapeHtml(lang)}` : "";

  return `
<article class="result-card" aria-label="Result ${rank}: ${symbol}">
  <div class="result-header">
    <span class="result-rank" aria-label="Rank">#${rank}</span>
    <span class="result-symbol" title="${symbol}">${symbol}</span>
    <span class="badge badge-lang" title="Language">${escapeHtml(lang)}</span>
    <span class="badge badge-type" title="Chunk type">${escapeHtml(type)}</span>
    <span class="badge badge-score" title="Fused score">${pct(r.score)}</span>
    <span class="match-label ${tier}" title="Relevance tier">${label}</span>
    <span class="result-filepath" title="${filepath}">${filepath}</span>
  </div>

  <div class="result-scores" aria-label="Score breakdown">
    <div class="score-bar-group" title="Overall fused relevance">
      <span class="score-bar-label">overall</span>
      <div class="score-bar-track">
        <div class="score-bar-fill ${tier}" style="width: ${fusedPct}%;"></div>
      </div>
      <span class="score-bar-pct">${fusedPct}%</span>
    </div>
    <div class="score-bar-group" title="Semantic embedding similarity">
      <span class="score-bar-label">sem</span>
      <div class="score-bar-track">
        <div class="score-bar-fill ${semTier}" style="width: ${semPct}%;"></div>
      </div>
      <span class="score-bar-pct">${semPct}%</span>
    </div>
    <div class="score-bar-group" title="BM25 keyword search score">
      <span class="score-bar-label">bm25</span>
      <div class="score-bar-track">
        <div class="score-bar-fill ${bm25Tier}" style="width: ${bm25Pct}%;"></div>
      </div>
      <span class="score-bar-pct">${bm25Pct}%</span>
    </div>
    <div class="score-bar-group" title="Exact symbol name match score">
      <span class="score-bar-label">sym</span>
      <div class="score-bar-track">
        <div class="score-bar-fill ${symTier}" style="width: ${symPct}%;"></div>
      </div>
      <span class="score-bar-pct">${symPct}%</span>
    </div>
  </div>

  <div class="code-wrapper">
    <div class="line-numbers" aria-hidden="true">${lineNumbers}</div>
    <pre class="result-code" tabindex="0" aria-label="Source code"><code class="${hljsLang}">${code}</code></pre>
  </div>

  <div class="result-footer">
    <span class="result-lines">${lines}</span>
    ${repo ? `<span class="badge badge-lang" title="Repository">${repo}</span>` : ""}
    <button class="btn btn-secondary copy-btn"
            style="font-size:0.75rem;padding:0.25rem 0.6rem"
            onclick="copyCode(this)" aria-label="Copy code to clipboard">Copy</button>
  </div>
</article>`;
}

window.copyCode = async function copyCode(btn) {
  const code = btn.closest(".result-card")?.querySelector("pre.result-code code")?.textContent ?? "";
  try {
    await navigator.clipboard.writeText(code);
    btn.classList.add("copied");
    btn.textContent = "Copied ✓";
    showToast("Code copied to clipboard", "success", 2000);
    setTimeout(() => {
      btn.classList.remove("copied");
      btn.textContent = "Copy";
    }, 1800);
  } catch (_) {
    btn.textContent = "Failed";
    showToast("Could not copy code", "error", 2000);
    setTimeout(() => { btn.textContent = "Copy"; }, 1500);
  }
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

let _pollTimer = null;

function startProgressPolling() {
  stopProgressPolling();
  _pollTimer = setInterval(async () => {
    try {
      const s = await apiFetch("/api/index/status");
      updateProgressUI(s);
    } catch (_) { /* server might briefly be busy — ignore */ }
  }, 800);
}

function stopProgressPolling() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

function updateProgressUI(s) {
  if (!s.active && s.stage !== "done") return;

  const stageLabels = {
    cloning:    "Cloning repository…",
    clearing:   "Clearing old index…",
    extracting: `Extracting files… ${s.files_done}/${s.files_total}`,
    embedding:  `Embedding chunks… ${s.chunks_done}/${s.chunks_total}`,
    indexing:   "Inserting into index…",
    done:       "Complete!",
  };
  const label = stageLabels[s.stage] || s.message || s.stage;
  if (progressLabel) progressLabel.textContent = label;

  if (s.stage === "embedding" && s.chunks_total > 0) {
    progressFill.className = "progress-fill";
    progressFill.style.width = `${s.pct}%`;
  }
}

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
  progressFill.style.width = "";
  if (progressLabel) progressLabel.textContent = "Starting…";

  startProgressPolling();

  try {
    const data = await apiFetch("/api/index", { method: "POST", body: JSON.stringify(body) });

    stopProgressPolling();
    progressFill.className   = "progress-fill";
    progressFill.style.width = "100%";
    if (progressLabel) progressLabel.textContent = `Done — ${data.chunks_indexed.toLocaleString()} chunks indexed`;

    istatRepo.textContent      = data.repo_name;
    istatChunks.textContent    = data.chunks_indexed.toLocaleString();
    istatFiles.textContent     = data.files_processed.toLocaleString();
    istatSkipped.textContent   = data.skipped_files.toLocaleString();
    if (istatUnchanged) {
      istatUnchanged.textContent = (data.skipped_unchanged ?? 0).toLocaleString();
    }
    istatDuration.textContent  = `${data.duration_seconds}s`;
    indexStats.hidden = false;

    const unchMsg = data.skipped_unchanged ? ` (${data.skipped_unchanged} unchanged files skipped)` : "";
    showAlert(indexAlert,
      `Indexed ${data.chunks_indexed.toLocaleString()} chunks from "${data.repo_name}" in ${data.duration_seconds}s${unchMsg}.`,
      "success");
    showToast(`Repository "${data.repo_name}" indexed successfully!`, "success");

    checkHealth(true);
    loadRepos();
    refreshRepoDropdown();
  } catch (err) {
    stopProgressPolling();
    progressFill.className   = "progress-fill";
    progressFill.style.width = "0%";
    if (progressLabel) progressLabel.textContent = "";
    showAlert(indexAlert, `Indexing failed: ${err.message}`, "error");
    showToast(`Indexing failed: ${err.message}`, "error");
  } finally {
    indexBtn.disabled     = false;
    indexBtn.textContent  = "Index repository";
    setTimeout(() => {
      indexProgress.hidden = true;
      if (progressLabel) progressLabel.textContent = "";
    }, 2000);
  }
}

indexBtn.addEventListener("click", runIndex);


// ── Repo Management ───────────────────────────────────────────────────────────

async function loadRepos() {
  repoListEl.innerHTML = `<p style="color:var(--text-muted);font-size:0.85rem;">Loading…</p>`;
  clearAlert(manageAlert);

  try {
    const data = await apiFetch("/api/repos?include_size=true");

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
    showToast(`Removed repository "${repoName}"`, "success");

    loadRepos();
    refreshRepoDropdown();
    checkHealth(true);
  } catch (err) {
    if (row) { row.style.opacity = ""; row.style.pointerEvents = ""; }
    showAlert(manageAlert, `Failed to remove "${repoName}": ${err.message}`, "error");
    showToast(`Failed to remove "${repoName}"`, "error");
  }
};

refreshReposBtn.addEventListener("click", loadRepos);


// ── Initialization ────────────────────────────────────────────────────────────

initTheme();
initWelcomeBanner();
renderRecentSearches();
