"""CSS stylesheet for the HTML evaluation report.

Light and dark theme tokens (``data-theme`` on the root element), the scorecard
matrix, the detail drawer, a phone layout and a print layout. No external
fonts or images: the report opens offline from ``file://``.
"""

CSS_CONTENT = r"""
:root{
  --bg:#f6f7f9;--panel:#fff;--ink:#0b0e13;--ink2:#4a5566;--ink3:#7a8596;--line:#dfe3ea;--line2:#edf0f4;
  --accent:#2f5bd3;--amber:#c8692f;
  --ok:#0f7a5c;--ok-bg:#e3f3ee;--warn:#8a5a00;--warn-bg:#fbf0d9;--bad:#b4232c;--bad-bg:#fbe7e8;
  --shadow:0 10px 40px rgba(11,14,19,.18);
  --font:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
:root[data-theme=dark]{
  --bg:#0b0e13;--panel:#10151d;--ink:#eceff4;--ink2:#a9b3c3;--ink3:#7d889a;--line:#2d3849;--line2:#1d2532;
  --accent:#7ea6ff;--amber:#e9a57c;
  --ok:#4fd1ae;--ok-bg:#12302a;--warn:#f0b45a;--warn-bg:#352a12;--bad:#f2868a;--bad-bg:#38191c;
  --shadow:0 10px 40px rgba(0,0,0,.6);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 var(--font);-webkit-text-size-adjust:100%}
.wrap{max-width:1180px;margin:0 auto;padding:0 16px 64px}
button,input,select{font:inherit;color:inherit}
button{cursor:pointer}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.mono{font-family:var(--mono)}
.muted{color:var(--ink2)}.faint{color:var(--ink3)}
/* header */
header{display:flex;align-items:center;gap:16px;padding:20px 0 16px;flex-wrap:wrap}
.logo{height:22px;display:block}.logo svg{height:22px;width:auto;display:block}
.logo-d{display:none}:root[data-theme=dark] .logo-l{display:none}:root[data-theme=dark] .logo-d{display:block}
.tag{font-size:13px;color:var(--ink2);border-left:1px solid var(--line);padding-left:16px}
.spacer{flex:1}
.tbtn{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:5px 11px;font-size:12px;color:var(--ink2)}
.meta{display:flex;gap:6px 22px;flex-wrap:wrap;color:var(--ink2);font-size:13px;padding-bottom:22px;border-bottom:1px solid var(--line)}
.meta b{color:var(--ink);font-weight:600}
/* verdict */
.verdict{padding:34px 0 8px;display:grid;grid-template-columns:1fr minmax(220px,340px);gap:24px 48px;align-items:end}
.vlabel{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);margin-bottom:6px}
.vmain{font-size:clamp(26px,4.4vw,40px);line-height:1.1;font-weight:650;letter-spacing:-.01em;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.vbadge{font-size:13px;font-weight:600;letter-spacing:.04em;padding:4px 10px;border-radius:6px}
.vbadge.bad{background:var(--bad-bg);color:var(--bad)}.vbadge.ok{background:var(--ok-bg);color:var(--ok)}
.vsub{color:var(--ink2);margin-top:8px;max-width:60ch}
.gate svg{width:100%;height:46px;display:block}
.gatecap{display:flex;justify-content:space-between;font-size:12px;color:var(--ink3);margin-top:2px}
/* kpi */
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin:30px 0 8px}
.kpi{padding:2px 20px 2px 0}
.kpi+.kpi{padding-left:20px;border-left:1px solid var(--line2)}
.kl{font-size:12px;color:var(--ink3);letter-spacing:.04em;text-transform:uppercase}
.kv{font-size:28px;font-weight:600;letter-spacing:-.01em;line-height:1.25;font-variant-numeric:tabular-nums}
.kv small{font-size:14px;font-weight:500;color:var(--ink2)}
.kd{font-size:12.5px;color:var(--ink2);font-variant-numeric:tabular-nums}
.kd .good{color:var(--ok)}.kd .badc{color:var(--bad)}.kd .flat{color:var(--ink3)}
.ks{font-size:12.5px;color:var(--ink3);margin-top:2px;font-variant-numeric:tabular-nums}
/* regression line */
.reg{margin:26px 0 0;padding:14px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);display:flex;gap:10px 22px;flex-wrap:wrap;align-items:center;font-size:13px}
.reg .lead{color:var(--ink2)}
.chip{display:inline-flex;gap:6px;align-items:center;border:1px solid var(--line);background:var(--panel);border-radius:999px;padding:2px 10px;font-size:12.5px}
.chip.k-bad{border-color:var(--bad);color:var(--bad)}.chip.k-ok{border-color:var(--ok);color:var(--ok)}
.chip.k-neu{color:var(--ink2)}
.reg a,.link{color:var(--accent);background:none;border:0;padding:0;text-decoration:underline;text-underline-offset:2px}
/* sections */
h2{font-size:17px;font-weight:600;margin:0;letter-spacing:-.005em}
.sec{margin-top:44px}
.sech{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.sech p{margin:0;color:var(--ink2);font-size:13px}
/* controls */
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:14px 0 10px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--panel)}
.seg button{border:0;background:none;padding:6px 12px;font-size:13px;color:var(--ink2);border-right:1px solid var(--line)}
.seg button:last-child{border-right:0}
.seg button[aria-pressed=true]{background:var(--ink);color:var(--bg)}
.seg .n{opacity:.65;margin-left:4px;font-variant-numeric:tabular-nums}
.ctl{border:1px solid var(--line);background:var(--panel);border-radius:8px;padding:6px 10px;font-size:13px}
input.ctl{min-width:190px}
.legend{display:flex;gap:6px 16px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:0 0 8px;align-items:center}
/* matrix */
.mwrap{overflow-x:auto;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:var(--panel)}
table.m{border-collapse:separate;border-spacing:0;width:100%;min-width:780px}
.m th,.m td{padding:0;text-align:left;vertical-align:middle}
.m thead th{font-size:12px;font-weight:600;color:var(--ink2);padding:12px 8px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:center;background:var(--panel)}
.m thead th.cn{text-align:left;padding-left:14px}
.m thead .thr{display:block;font-weight:400;color:var(--ink3);font-size:11px;font-variant-numeric:tabular-nums}
.m tbody tr.r{cursor:pointer}
.m tbody tr.r:hover td,.m tbody tr.r:focus-visible td{background:var(--line2)}
.m tbody td{border-bottom:1px solid var(--line2);height:58px;background:var(--panel)}
.m td.cn{padding:6px 10px 6px 14px;min-width:230px;max-width:300px}
.cname{font-weight:600;display:flex;gap:8px;align-items:center}
.cname .t{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cid{font-size:12px;color:var(--ink3)}
.kind{font-size:10.5px;letter-spacing:.06em;font-weight:600;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--ink2)}
.kind.sim{border-style:dashed}
.st{font-size:15px;width:18px;text-align:center;flex:none}
.st.pass{color:var(--ok)}.st.fail{color:var(--bad)}.st.err{color:var(--warn)}
td.c{padding:5px 4px;text-align:center}
.cell{position:relative;display:inline-flex;flex-direction:column;align-items:center;justify-content:center;width:68px;height:44px;border-radius:7px;font-variant-numeric:tabular-nums;line-height:1.15}
.cell .g{font-size:13px;font-weight:700}.cell .s{font-size:12px}
.cell.pass{background:var(--ok-bg);color:var(--ok)}
.cell.near{background:var(--warn-bg);color:var(--warn);outline:1.5px dashed var(--warn);outline-offset:-2px}
.cell.fail{background:var(--bad-bg);color:var(--bad);outline:1.5px solid var(--bad);outline-offset:-2px}
.cell.na{color:var(--ink3);background:none}
.cell .chg{position:absolute;top:2px;right:3px;font-size:9px;font-weight:700;line-height:1}
.cell .chg.up{color:var(--ok)}.cell .chg.dn{color:var(--bad)}.cell .chg.nw{color:var(--accent);font-size:8.5px;letter-spacing:.02em}
.cell.chgd::after{content:"";position:absolute;left:-1px;top:-1px;width:0;height:0;border-style:solid;border-width:9px 9px 0 0;border-color:var(--accent) transparent transparent transparent;border-top-left-radius:7px}
td.tot{padding:6px 14px 6px 10px;text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums;min-width:110px}
.tot .a{font-weight:600}.tot .b{font-size:12px;color:var(--ink3)}
td.rg{padding:6px 14px 6px 4px;font-size:12px;white-space:nowrap;min-width:106px}
.rgl{font-weight:600}.rgl.bad{color:var(--bad)}.rgl.ok{color:var(--ok)}.rgl.neu{color:var(--accent)}.rgl.flat{color:var(--ink3);font-weight:400}
.m tfoot td{padding:10px 4px;text-align:center;font-size:12.5px;font-variant-numeric:tabular-nums;border-top:1px solid var(--line);background:var(--panel)}
.m tfoot td.cn{text-align:left;padding-left:14px;font-weight:600;color:var(--ink2)}
.m tfoot .pv{display:block;font-size:11px;color:var(--ink3)}
.empty{padding:28px;text-align:center;color:var(--ink3)}
/* run over run */
.ror{display:grid;grid-template-columns:minmax(150px,230px) 1fr 150px;gap:10px 18px;align-items:center}
.ror .nm{font-size:13px;font-weight:600;overflow:hidden;text-overflow:ellipsis}
.ror .nm small{display:block;font-weight:400;color:var(--ink3);font-size:11.5px}
.ror svg{width:100%;height:34px;display:block}
.ror .dl{font-size:12.5px;font-variant-numeric:tabular-nums;color:var(--ink2);text-align:right}
.key{display:flex;gap:16px;font-size:12px;color:var(--ink2);margin:0 0 10px;align-items:center;flex-wrap:wrap}
.sw{display:inline-block;width:14px;height:8px;border-radius:2px;vertical-align:middle;margin-right:5px}
/* strip */
.strip{display:grid;grid-template-columns:minmax(120px,200px) 1fr 1fr;gap:6px 18px;align-items:center}
.strip .h{font-size:11.5px;color:var(--ink3);letter-spacing:.04em;text-transform:uppercase}
.strip .nm{font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar{display:flex;align-items:center;gap:8px;height:20px;font-size:12px;font-variant-numeric:tabular-nums;color:var(--ink2)}
.bar i{display:block;height:10px;border-radius:2px;background:var(--accent);min-width:2px}
.bar i.sim{background:repeating-linear-gradient(135deg,var(--amber) 0 3px,transparent 3px 6px);border:1px solid var(--amber)}
/* details (collapsed) */
details.fold{border-top:1px solid var(--line);padding:12px 0}
details.fold:last-of-type{border-bottom:1px solid var(--line)}
details.fold>summary{cursor:pointer;font-weight:600;font-size:14px;list-style:none;display:flex;gap:10px;align-items:baseline}
details.fold>summary::-webkit-details-marker{display:none}
details.fold>summary::before{content:"+";width:14px;color:var(--ink3);font-weight:400}
details.fold[open]>summary::before{content:"-"}
details.fold>summary span{font-weight:400;color:var(--ink3);font-size:12.5px}
.fb{padding:14px 0 4px}
table.t{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
.t th{font-weight:600;color:var(--ink2);text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--line)}
.t td{padding:6px 10px 6px 0;border-bottom:1px solid var(--line2);vertical-align:top}
.t .num{text-align:right}
.scrollx{overflow-x:auto}
pre,.code{font:12px/1.5 var(--mono);background:var(--bg);border:1px solid var(--line2);border-radius:6px;padding:10px 12px;margin:0;white-space:pre-wrap;word-break:break-word;max-height:300px;overflow:auto}
/* drawer */
.scrim{position:fixed;inset:0;background:rgba(5,8,12,.45);opacity:0;pointer-events:none;transition:opacity .2s;z-index:40}
.scrim.on{opacity:1;pointer-events:auto}
aside.dr{position:fixed;top:0;right:0;bottom:0;width:min(720px,100%);background:var(--panel);box-shadow:var(--shadow);transform:translateX(102%);transition:transform .22s ease;z-index:50;display:flex;flex-direction:column;border-left:1px solid var(--line)}
aside.dr.on{transform:none}
.drh{padding:18px 22px 14px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:flex-start}
.drh h3{margin:0;font-size:18px;line-height:1.3;font-weight:600}
.drh .x{margin-left:auto;border:1px solid var(--line);background:none;border-radius:8px;width:34px;height:34px;font-size:16px;flex:none}
.drn{display:flex;gap:6px;flex:none}
.drn button{border:1px solid var(--line);background:none;border-radius:8px;width:34px;height:34px}
.drb{overflow:auto;padding:6px 22px 40px;flex:1}
.dsec{padding:18px 0 4px}
.dsec>h4{margin:0 0 8px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink3);font-weight:600}
.trio{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.trio h5{margin:0 0 4px;font-size:12px;color:var(--ink2);font-weight:600}
.trio .tx{font-size:13px;white-space:pre-wrap;word-break:break-word}
.crit{padding:12px 0;border-bottom:1px solid var(--line2)}
.crit:last-child{border-bottom:0}
.crit .top{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}
.crit .nm{font-weight:600}
.pill{font-size:11.5px;font-weight:600;border-radius:5px;padding:1px 7px}
.pill.pass{background:var(--ok-bg);color:var(--ok)}.pill.fail{background:var(--bad-bg);color:var(--bad)}.pill.err{background:var(--warn-bg);color:var(--warn)}
.sb svg{width:100%;height:18px;display:block;margin:6px 0 2px}
.reason{font-size:13px;color:var(--ink2)}
.tag2{font-size:11.5px;color:var(--ink3)}
.tr{display:flex;flex-direction:column;gap:10px}
.msg{max-width:88%;padding:8px 12px;border-radius:10px;font-size:13px;white-space:pre-wrap;word-break:break-word}
.msg.u{align-self:flex-end;background:var(--line2);border-bottom-right-radius:3px}
.msg.a{align-self:flex-start;border:1px solid var(--line);border-bottom-left-radius:3px}
.msg small{display:block;font-size:11px;color:var(--ink3);margin-bottom:2px;font-weight:600}
.goal{display:flex;gap:8px;font-size:13px;padding:3px 0}
.goal .gi{width:18px;flex:none;font-weight:700}
.goal .gi.y{color:var(--ok)}.goal .gi.n{color:var(--bad)}.goal .gi.q{color:var(--ink3)}
.steps{margin:0;padding:0;list-style:none;font-size:13px}
.steps li{display:flex;gap:10px;padding:5px 0;border-bottom:1px solid var(--line2);align-items:baseline;flex-wrap:wrap}
.steps .k{font-size:11px;border:1px solid var(--line);border-radius:4px;padding:0 5px;color:var(--ink2);flex:none}
.note{font-size:12.5px;color:var(--warn);background:var(--warn-bg);border-radius:6px;padding:6px 10px;margin:6px 0}
details.sub{border-top:1px solid var(--line2);padding:8px 0}
details.sub>summary{cursor:pointer;font-size:13px;color:var(--ink2)}
.kvg{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.kvg div b{display:block;font-size:17px;font-variant-numeric:tabular-nums}
.kvg div span{font-size:11.5px;color:var(--ink3)}
@media(max-width:760px){
  .verdict{grid-template-columns:1fr}
  .kpis{grid-template-columns:1fr 1fr;row-gap:18px}
  .kpi:nth-child(3){padding-left:0;border-left:0}
  .ror{grid-template-columns:1fr;gap:2px}.ror .dl{text-align:left;margin-bottom:12px}
  .strip{grid-template-columns:1fr}.strip .h.sp{display:none}
  .strip .nm{margin-top:10px;font-weight:600}
  .trio{grid-template-columns:1fr}
  .tag{display:none}
  input.ctl{flex:1;min-width:120px}
}
@media print{
  body{background:#fff}.tbtn,.controls,.scrim,aside.dr{display:none!important}
  .mwrap{overflow:visible}details.fold>*{display:block}
  .wrap{max-width:none}
}
"""
