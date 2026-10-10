"""HTML page structure for the evaluation report.

``build_html`` assembles CSS, JS, and case items into a
self-contained HTML document (no external CDN dependencies).
"""

from __future__ import annotations


_EXTRA_CSS = """
/* 10xGraph report polish: tokens, typography, layout overrides */
:root {
    --color-bg: #f6f7f9;
    --color-card: #ffffff;
    --color-border: #dfe3e8;
    --color-text: #16202c;
    --color-muted: #566272;
    --color-header-bg: #ffffff;
    --color-bg-muted: #eef0f3;
    --color-accent: #3b4fd8;
    --color-pass: #15803d;
    --color-fail: #b91c1c;
    --color-warn: #b45309;
    --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        --color-bg: #0d1117;
        --color-card: #161b22;
        --color-border: #30363d;
        --color-text: #e6edf3;
        --color-muted: #9aa7b5;
        --color-header-bg: #161b22;
        --color-bg-muted: #0d1117;
        --color-accent: #8b9cff;
        --color-pass: #3fb950;
        --color-fail: #ff7b72;
        --color-warn: #e3b341;
    }
}
:root[data-theme="dark"] {
    --color-bg: #0d1117;
    --color-card: #161b22;
    --color-border: #30363d;
    --color-text: #e6edf3;
    --color-muted: #9aa7b5;
    --color-header-bg: #161b22;
    --color-bg-muted: #0d1117;
    --color-accent: #8b9cff;
    --color-pass: #3fb950;
    --color-fail: #ff7b72;
    --color-warn: #e3b341;
}
html { background: var(--color-bg); }
body {
    background-color: var(--color-bg);
    color: var(--color-text);
    font-family: var(--font-sans);
    line-height: 1.55;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
}
code, pre, .response-box, .score-value, .case-score, .tool-table td, .node-box p,
.error-message, .traj-label, .timestamp, .report-title-chip { font-family: var(--font-mono); }
.container { max-width: 1120px; padding: 0 1rem 2rem; }
.sticky-header { margin: 0 -1rem 1.5rem; padding: 0.7rem 1rem; box-shadow: none; }
.header-inner { flex-wrap: wrap; }
.brand-logo { display: none; }
.brand-name, .footer-brand-name {
    background: none; -webkit-text-fill-color: currentColor; color: var(--color-text);
    font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em;
}
.brand-sub { font-size: 0.68rem; }
.report-title-chip {
    font-size: 0.8rem; color: var(--color-text); background: var(--color-bg-muted);
    max-width: min(60vw, 360px);
}
.header-meta { flex-wrap: wrap; gap: 0.5rem; flex-shrink: 1; }
.header-link:hover { background: var(--color-accent); border-color: var(--color-accent); color: var(--color-card); }
.header-link, .theme-btn, .filter-btn, .search-input { min-height: 32px; }
a:focus-visible, button:focus-visible, summary:focus-visible, .case-header:focus-visible {
    outline: 2px solid var(--color-accent); outline-offset: 2px;
}
.summary { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; }
.stat-card { text-align: left; border-radius: 8px; padding: 0.9rem 1rem; }
.stat-card:hover { transform: none; box-shadow: none; }
.stat-value { font-size: 1.75rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.stat-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
.stat-rate .stat-value { background: none; -webkit-text-fill-color: currentColor; color: var(--color-accent); }
.progress-fill { background: var(--color-accent); }
.token-summary-bar {
    grid-column: 1 / -1; font-size: 0.85rem; color: var(--color-muted);
    background: var(--color-card); border: 1px solid var(--color-border);
    border-radius: 8px; padding: 0.6rem 1rem; overflow-wrap: anywhere;
}
.chart-panel { border-radius: 8px; box-shadow: none; overflow-x: auto; }
.cases-section { min-width: 0; }
.filter-bar { top: 0; position: static; }
.search-input { flex: 1 1 180px; max-width: 320px; }
.filter-btn { border-radius: 6px; }
.case-item { border-radius: 8px; min-width: 0; }
.case-item:hover { box-shadow: none; transform: none; border-color: var(--color-accent); }
.case-header { flex-wrap: wrap; }
.case-name { min-width: 0; overflow-wrap: anywhere; }
.case-status { color: var(--color-card); }
.case-details { min-width: 0; }
.detail-section summary { padding: 0.4rem 0; }
.response-box, .node-box, .error-message { overflow-wrap: anywhere; }
.tool-table { display: block; max-width: 100%; overflow-x: auto; }
.criterion-row { grid-template-columns: 1.25rem minmax(5rem, 9rem) 1fr auto 1.25rem; }
@media (max-width: 640px) {
    .header-inner { align-items: flex-start; }
    .criterion-row { grid-template-columns: 1.25rem 1fr auto; }
    .criterion-row .score-bar-wrap { grid-column: 1 / -1; order: 5; }
    .criterion-reason { grid-column: 1 / -1; }
    .search-input { max-width: none; margin-left: 0; }
}
.footer-link { color: var(--color-accent); }
.footer-link:hover { color: var(--color-accent); }
.footer-logo { display: none; }
"""


def build_html(
    *,
    title: str,
    timestamp: str,
    total_cases: int,
    passed_cases: int,
    failed_cases: int,
    error_cases: int,
    pass_rate_pct: str,
    duration: str,
    case_items: str,
    css: str,
    js: str,
    token_summary: str = "",
) -> str:
    """Return a complete self-contained HTML string for an eval report."""
    token_summary_block = (
        f'        <div class="token-summary-bar">{token_summary}</div>' if token_summary else ""
    )
    extra_css = _EXTRA_CSS
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css}
{extra_css}
    </style>
</head>
<body>
    <div class="container">

        <header class="sticky-header">
            <div class="header-inner">
                <div class="brand">
                    <div class="brand-text">
                        <span class="brand-name">10xGraph</span>
                        <span class="brand-sub">Evaluation Report</span>
                    </div>
                    <span class="report-title-chip" title="{title}">{title}</span>
                </div>
                <div class="header-meta">
                    <span class="timestamp">Generated: {timestamp}</span>
                    <a class="header-link" href="https://10xgraph.com" target="_blank" rel="noopener" title="Documentation">Docs</a>
                    <a class="header-link" href="https://github.com/10xGraph/10xGraph" target="_blank" rel="noopener" title="GitHub Repository">GitHub</a>
                    <button id="theme-toggle" class="theme-btn" aria-label="Toggle dark mode"></button>
                </div>
            </div>
        </header>

        <section class="summary" aria-label="Summary">
            <div class="stat-card">
                <div class="stat-value">{total_cases}</div>
                <div class="stat-label">Total</div>
            </div>
            <div class="stat-card stat-pass">
                <div class="stat-value">{passed_cases}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card stat-fail">
                <div class="stat-value">{failed_cases}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card stat-warn">
                <div class="stat-value">{error_cases}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-card stat-rate">
                <div class="stat-value">{pass_rate_pct}%</div>
                <div class="stat-label">Pass rate</div>
                <div class="progress-bar" role="progressbar" aria-label="Pass rate" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{pass_rate_pct}">
                    <div class="progress-fill" style="width: {pass_rate_pct}%"></div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{duration}s</div>
                <div class="stat-label">Duration</div>
            </div>
{token_summary_block}
        </section>

        <section class="charts-section">
            <div class="chart-panel">
                <h2>Criterion breakdown</h2>
                <div id="criterion-breakdown"></div>
            </div>
            <div class="chart-panel">
                <h2>Score by case</h2>
                <div id="case-chart"></div>
            </div>
        </section>

        <section class="cases-section">
            <div class="filter-bar">
                <button class="filter-btn active" data-filter="all">All</button>
                <button class="filter-btn" data-filter="pass">Passed</button>
                <button class="filter-btn" data-filter="fail">Failed</button>
                <button class="filter-btn" data-filter="error">Errors</button>
                <input id="case-search" class="search-input" type="search"
                       placeholder="Search cases" aria-label="Search cases" />
            </div>
            <div class="case-list">
{case_items}
            </div>
        </section>

        <footer>
            <div class="footer-inner">
                <div class="footer-brand">
                    <span class="footer-brand-name">10xGraph</span>
                    <span class="footer-tagline">Graph engineering for production AI agents</span>
                </div>
                <div class="footer-links">
                    <a class="footer-link" href="https://10xgraph.com" target="_blank" rel="noopener">Documentation</a>
                    <span class="footer-sep" aria-hidden="true">|</span>
                    <a class="footer-link" href="https://github.com/10xGraph/10xGraph" target="_blank" rel="noopener">GitHub</a>
                    <span class="footer-sep" aria-hidden="true">|</span>
                    <a class="footer-link" href="https://pypi.org/project/10xgraph/" target="_blank" rel="noopener">PyPI</a>
                </div>
                <p class="footer-note">Generated by 10xGraph. This report is self-contained and needs no internet connection to view.</p>
            </div>
        </footer>

    </div>
    <script>
{js}
    </script>
</body>
</html>"""
