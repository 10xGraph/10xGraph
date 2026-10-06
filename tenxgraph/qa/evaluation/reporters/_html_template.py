"""HTML page structure for the evaluation report.

``build_html`` assembles CSS, JS, and case items into a
self-contained HTML document (no external CDN dependencies).
"""

from __future__ import annotations


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
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
{css}
    </style>
</head>
<body>
    <div class="container">

        <header class="sticky-header">
            <div class="header-inner">
                <div class="brand">
                    <span class="brand-logo" aria-hidden="true">&#x26A1;</span>
                    <div class="brand-text">
                        <span class="brand-name">AgentFlow</span>
                        <span class="brand-sub">Evaluation Report</span>
                    </div>
                    <span class="report-title-chip">{title}</span>
                </div>
                <div class="header-meta">
                    <span class="timestamp">Generated: {timestamp}</span>
                    <a class="header-link" href="https://agentflow.10xscale.ai/" target="_blank" rel="noopener" title="Documentation">
                        &#x1F4D6; Docs
                    </a>
                    <a class="header-link" href="https://github.com/10xHub/Agentflow" target="_blank" rel="noopener" title="GitHub Repository">
                        &#x1F4BE; GitHub
                    </a>
                    <button id="theme-toggle" class="theme-btn" aria-label="Toggle dark mode">&#x1F319;</button>
                </div>
            </div>
        </header>

        <section class="summary">
            <div class="stat-card">
                <div class="stat-icon">&#x1F4CB;</div>
                <div class="stat-value">{total_cases}</div>
                <div class="stat-label">Total Cases</div>
            </div>
            <div class="stat-card stat-pass">
                <div class="stat-icon">&#x2705;</div>
                <div class="stat-value">{passed_cases}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card stat-fail">
                <div class="stat-icon">&#x274C;</div>
                <div class="stat-value">{failed_cases}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card stat-warn">
                <div class="stat-icon">&#x26A0;&#xFE0F;</div>
                <div class="stat-value">{error_cases}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-card stat-rate">
                <div class="stat-icon">&#x1F4C8;</div>
                <div class="stat-value">{pass_rate_pct}%</div>
                <div class="stat-label">Pass Rate</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {pass_rate_pct}%"></div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">&#x23F1;&#xFE0F;</div>
                <div class="stat-value">{duration}s</div>
                <div class="stat-label">Duration</div>
            </div>
{token_summary_block}
        </section>

        <section class="charts-section">
            <div class="chart-panel">
                <h2>&#x1F4CA; Criterion Breakdown</h2>
                <div id="criterion-breakdown"></div>
            </div>
            <div class="chart-panel">
                <h2>&#x1F3AF; Score by Case</h2>
                <div id="case-chart"></div>
            </div>
        </section>

        <section class="cases-section">
            <div class="filter-bar">
                <button class="filter-btn active" data-filter="all">All</button>
                <button class="filter-btn" data-filter="pass">&#x2705; Passed</button>
                <button class="filter-btn" data-filter="fail">&#x274C; Failed</button>
                <button class="filter-btn" data-filter="error">&#x26A0; Errors</button>
                <input id="case-search" class="search-input" type="search"
                       placeholder="&#x1F50D; Search cases&#x2026;" />
            </div>
            <div class="case-list">
{case_items}
            </div>
        </section>

        <footer>
            <div class="footer-inner">
                <div class="footer-brand">
                    <span class="footer-logo" aria-hidden="true">&#x26A1;</span>
                    <span class="footer-brand-name">AgentFlow</span>
                    <span class="footer-tagline">— Multi-Agent AI Framework by 10xScale</span>
                </div>
                <div class="footer-links">
                    <a class="footer-link" href="https://agentflow.10xscale.ai/" target="_blank" rel="noopener">
                        &#x1F4D6; Documentation
                    </a>
                    <span class="footer-sep" aria-hidden="true">&#x2022;</span>
                    <a class="footer-link" href="https://github.com/10xHub/Agentflow" target="_blank" rel="noopener">
                        &#x1F4BE; GitHub
                    </a>
                    <span class="footer-sep" aria-hidden="true">&#x2022;</span>
                    <a class="footer-link" href="https://pypi.org/project/10xscale-agentflow/" target="_blank" rel="noopener">
                        &#x1F4E6; PyPI
                    </a>
                </div>
                <p class="footer-note">Generated by the AgentFlow Evaluation Framework &nbsp;&middot;&nbsp; Report is self-contained, no internet connection required to view.</p>
            </div>
        </footer>

    </div>
    <script>
{js}
    </script>
</body>
</html>"""
