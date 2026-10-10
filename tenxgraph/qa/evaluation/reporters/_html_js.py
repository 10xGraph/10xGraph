"""JavaScript for the HTML evaluation report.

Provides:
  - Dark mode toggle (respects prefers-color-scheme, toggleable)
  - Case expand / collapse
  - Status filter buttons
  - Case name search
  - Inline SVG bar charts (criterion breakdown + score by case)
    — No external CDN; pure DOM-based SVG rendering (~100 lines)
"""

JS_CONTENT = r"""
(function () {
    'use strict';

    /* ── Dark mode ─────────────────────────────────────────────────────── */
    const root = document.documentElement;
    const themeBtn = document.getElementById('theme-toggle');
    const mq = window.matchMedia('(prefers-color-scheme: dark)');

    function applyTheme(dark) {
        root.setAttribute('data-theme', dark ? 'dark' : 'light');
        if (themeBtn) { themeBtn.textContent = dark ? '\u2600\ufe0f' : '\ud83c\udf19'; }
    }
    applyTheme(mq.matches);
    if (themeBtn) {
        themeBtn.addEventListener('click', function () {
            applyTheme(root.getAttribute('data-theme') !== 'dark');
        });
    }

    /* ── Case expand / collapse ────────────────────────────────────────── */
    document.querySelectorAll('.case-header').forEach(function (header) {
        header.addEventListener('click', function () {
            header.closest('.case-item').classList.toggle('expanded');
        });
    });

    /* ── Filter buttons ────────────────────────────────────────────────── */
    var activeFilter = 'all';
    document.querySelectorAll('.filter-btn').forEach(function (btn) {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.filter-btn').forEach(function (b) {
                b.classList.remove('active');
            });
            btn.classList.add('active');
            activeFilter = btn.dataset.filter;
            applyFilters();
        });
    });

    /* ── Search ────────────────────────────────────────────────────────── */
    var searchQuery = '';
    var searchInput = document.getElementById('case-search');
    if (searchInput) {
        searchInput.addEventListener('input', function () {
            searchQuery = searchInput.value.toLowerCase();
            applyFilters();
        });
    }

    function applyFilters() {
        document.querySelectorAll('.case-item').forEach(function (item) {
            var statusOk = activeFilter === 'all' || item.dataset.status === activeFilter;
            var nameEl = item.querySelector('.case-name');
            var name = nameEl ? nameEl.textContent.toLowerCase() : '';
            var searchOk = !searchQuery || name.indexOf(searchQuery) !== -1;
            item.style.display = (statusOk && searchOk) ? '' : 'none';
        });
        renderCaseChart();
    }

    /* ── SVG helpers ───────────────────────────────────────────────────── */
    var SVG_NS = 'http://www.w3.org/2000/svg';

    function makeSvg(w, h) {
        var svg = document.createElementNS(SVG_NS, 'svg');
        svg.setAttribute('width', w);
        svg.setAttribute('height', h);
        svg.setAttribute('class', 'chart-svg');
        svg.setAttribute('aria-hidden', 'true');
        return svg;
    }

    function makeRect(x, y, w, h, fill) {
        var r = document.createElementNS(SVG_NS, 'rect');
        r.setAttribute('x', x);
        r.setAttribute('y', y);
        r.setAttribute('width', Math.max(0, w));
        r.setAttribute('height', h);
        r.setAttribute('fill', fill);
        r.setAttribute('rx', 3);
        return r;
    }

    function makeText(x, y, txt, anchor, cls) {
        var t = document.createElementNS(SVG_NS, 'text');
        t.setAttribute('x', x);
        t.setAttribute('y', y);
        t.setAttribute('text-anchor', anchor);
        if (cls) { t.setAttribute('class', cls); }
        t.textContent = txt;
        return t;
    }

    function scoreColor(score) {
        var hue = Math.round(score * 120);
        return 'hsl(' + hue + ',65%,45%)';
    }

    function truncate(str, maxLen) {
        return str.length > maxLen ? str.slice(0, maxLen - 1) + '\u2026' : str;
    }

    /* ── Criterion breakdown chart ──────────────────────────────────────── */
    function renderCriterionChart() {
        var container = document.getElementById('criterion-breakdown');
        if (!container) { return; }
        container.innerHTML = '';

        var critMap = {};
        document.querySelectorAll('.criterion-row').forEach(function (row) {
            var name = row.dataset.criterion;
            var score = parseFloat(row.dataset.score);
            if (!name || isNaN(score)) { return; }
            if (!critMap[name]) { critMap[name] = { total: 0, count: 0 }; }
            critMap[name].total += score;
            critMap[name].count += 1;
        });

        var entries = Object.keys(critMap).map(function (name) {
            var d = critMap[name];
            return { name: name, avg: d.total / d.count };
        }).sort(function (a, b) { return b.avg - a.avg; });

        if (!entries.length) {
            container.textContent = 'No criteria data.';
            return;
        }

        var BAR_H = 22, GAP = 6;
        var PAD = { top: 8, left: 140, right: 48, bottom: 8 };
        var W = container.clientWidth || 260;
        var availW = W - PAD.left - PAD.right;
        var H = entries.length * (BAR_H + GAP) + PAD.top + PAD.bottom;
        var svg = makeSvg(W, H);

        entries.forEach(function (entry, i) {
            var y = PAD.top + i * (BAR_H + GAP);
            var barW = entry.avg * availW;
            /* track */
            var bg = makeRect(PAD.left, y, availW, BAR_H, 'var(--color-border)');
            bg.style.opacity = '0.3';
            svg.appendChild(bg);
            /* fill */
            svg.appendChild(makeRect(PAD.left, y, barW, BAR_H, scoreColor(entry.avg)));
            /* labels */
            svg.appendChild(makeText(PAD.left - 6, y + BAR_H * 0.68, truncate(entry.name, 20), 'end', 'chart-label'));
            svg.appendChild(makeText(PAD.left + barW + 5, y + BAR_H * 0.68, entry.avg.toFixed(2), 'start', 'chart-val'));
        });

        container.appendChild(svg);
    }

    /* ── Score by case chart ────────────────────────────────────────────── */
    function renderCaseChart() {
        var container = document.getElementById('case-chart');
        if (!container) { return; }
        container.innerHTML = '';

        var items = Array.prototype.slice.call(document.querySelectorAll('.case-item')).filter(function (el) {
            return el.style.display !== 'none';
        });

        if (!items.length) {
            container.textContent = 'No cases to display.';
            return;
        }

        var BAR_H = 18, GAP = 6;
        var PAD = { top: 8, left: 140, right: 48, bottom: 8 };
        var W = container.clientWidth || 260;
        var availW = W - PAD.left - PAD.right;
        var H = items.length * (BAR_H + GAP) + PAD.top + PAD.bottom;
        var svg = makeSvg(W, H);

        items.forEach(function (item, i) {
            var nameEl = item.querySelector('.case-name');
            var rawName = nameEl ? nameEl.textContent : '';
            var name = truncate(rawName, 22);
            var score = parseFloat(item.dataset.score || '0');
            var status = item.dataset.status;
            var y = PAD.top + i * (BAR_H + GAP);
            var barW = score * availW;
            var fill = status === 'pass'  ? 'var(--color-pass)'
                     : status === 'error' ? 'var(--color-warn)'
                     : 'var(--color-fail)';
            /* track */
            var bg = makeRect(PAD.left, y, availW, BAR_H, 'var(--color-border)');
            bg.style.opacity = '0.2';
            svg.appendChild(bg);
            /* fill */
            svg.appendChild(makeRect(PAD.left, y, barW, BAR_H, fill));
            /* labels */
            svg.appendChild(makeText(PAD.left - 6, y + BAR_H * 0.75, name, 'end', 'chart-label'));
            svg.appendChild(makeText(PAD.left + barW + 5, y + BAR_H * 0.75, score.toFixed(2), 'start', 'chart-val'));
        });

        container.appendChild(svg);
    }

    /* ── Initial render ─────────────────────────────────────────────────── */
    renderCriterionChart();
    renderCaseChart();

    /* Re-render on resize */
    var _rTimer;
    window.addEventListener('resize', function () {
        clearTimeout(_rTimer);
        _rTimer = setTimeout(function () {
            renderCriterionChart();
            renderCaseChart();
        }, 150);
    });

}());
"""
