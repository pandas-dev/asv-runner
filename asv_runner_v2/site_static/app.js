"use strict";

const COMMIT_URL = "https://github.com/pandas-dev/pandas/commit/";
const COMPARE_URL = "https://github.com/pandas-dev/pandas/compare/";
const PR_URL = "https://github.com/pandas-dev/pandas/pull/";
const MAX_SERIES = 64;
const DOUBLE_CLICK_MS = 250;
const CHART_HEIGHT = 420;

const RECENT_DAYS = 30;
const GROUP_LIMIT = 10; // detections listed per commit before "show more"

let INDEX = null;
const envByCommit = new Map();
// Detections from index.json, by commit position: [{name, params, kind, step}]
const detByCommit = new Map();
// Benchmarks with a regression in the last RECENT_DAYS days.
const recentRegressed = new Set();
const graphCache = new Map();
const state = {
  name: null,
  graph: null,
  selected: [],
  plot: null,
  shownIndex: new Map(),
  // Zoomed range and selected point, as positions in INDEX.commits.
  zoom: null, // {from, to} or null
  point: null,
  // Series hidden from the legend, keyed by seriesKey().
  hidden: new Set(),
};

// A series' parameter values as JSON: robust to commas and spaces in values.
function seriesKey(params) {
  return JSON.stringify(params);
}

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---- formatting ------------------------------------------------------------

function fmtVal(v, unit) {
  if (v == null) return "";
  const a = Math.abs(v);
  if (unit === "seconds") {
    if (a === 0) return "0 s";
    if (a < 1e-6) return (v * 1e9).toPrecision(3) + " ns";
    if (a < 1e-3) return (v * 1e6).toPrecision(3) + " µs";
    if (a < 1) return (v * 1e3).toPrecision(3) + " ms";
    return v.toPrecision(3) + " s";
  }
  if (unit === "bytes") {
    if (a < 1e3) return v + " B";
    if (a < 1e6) return (v / 1e3).toPrecision(3) + " kB";
    if (a < 1e9) return (v / 1e6).toPrecision(3) + " MB";
    return (v / 1e9).toPrecision(3) + " GB";
  }
  return v.toPrecision(4) + (unit ? " " + unit : "");
}

function fmtDate(ms) {
  return new Date(ms).toISOString().slice(0, 16).replace("T", " ");
}

function shortSha(sha) {
  return sha.slice(0, 10);
}

// Commit title without its "(#12345)" suffix; the PR is linked separately.
function commitTitle(i) {
  const t = INDEX.commits.title?.[i];
  return t ? t.replace(/\s*\(#\d+\)\s*$/, "") : null;
}

function prLink(i) {
  const pr = INDEX.commits.pr?.[i];
  return pr ? el("a", { href: PR_URL + pr, target: "_blank" }, `#${pr}`) : "";
}

// "Title (#123)", or nothing when the title is not stored.
function titleNode(i) {
  const t = commitTitle(i);
  if (!t) return "";
  const pr = prLink(i);
  return el("span", { class: "commit-title" }, t, pr ? " (" : "", pr, pr ? ")" : "");
}

// ---- URL fragment ----------------------------------------------------------
// Same format as asv and the links in existing issues:
//   #<benchmark>?p-<param>=<value>&p-<param>=<value>...
// A parameter may repeat to select several values; missing parameters mean all.

function parseHash() {
  const h = location.hash.slice(1);
  if (!h) return null;
  const q = h.indexOf("?");
  const name = decodeURIComponent(q < 0 ? h : h.slice(0, q));
  const params = new Map();
  const opts = new Map(); // view options: log, x, env, envall, from, to, at
  const hidden = []; // repeated hide= entries
  if (q >= 0) {
    for (const part of h.slice(q + 1).split("&")) {
      const eq = part.indexOf("=");
      if (eq < 0) continue;
      const value = decodeURIComponent(part.slice(eq + 1));
      if (part.startsWith("p-")) {
        const key = decodeURIComponent(part.slice(2, eq));
        if (!params.has(key)) params.set(key, new Set());
        params.get(key).add(value);
      } else if (part.startsWith("hide=")) {
        hidden.push(value);
      } else {
        opts.set(part.slice(0, eq), value);
      }
    }
  }
  return { name, params, opts, hidden };
}

function commitBySha(prefix) {
  if (!prefix) return null;
  const i = INDEX.commits.sha.findIndex((sha) => sha.startsWith(prefix.toLowerCase()));
  return i < 0 ? null : i;
}

function writeHash() {
  const g = state.graph;
  const parts = [];
  g.param_names.forEach((p, n) => {
    const all = paramValues(g, n);
    if (state.selected[n].size === all.length) return;
    for (const v of all) {
      if (state.selected[n].has(v)) parts.push(`p-${encodeURIComponent(p)}=${encodeURIComponent(v)}`);
    }
  });
  // View options after the parameters, so links keep the issue-link format.
  if ($("opt-log").checked) parts.push("log=1");
  if ($("opt-order").checked) parts.push("x=commit");
  if (!$("opt-env").checked) parts.push("env=0");
  if ($("opt-env-all").checked) parts.push("envall=1");
  if (state.zoom) {
    parts.push(`from=${shortSha(INDEX.commits.sha[state.zoom.from])}`);
    parts.push(`to=${shortSha(INDEX.commits.sha[state.zoom.to])}`);
  }
  if (state.point !== null) parts.push(`at=${shortSha(INDEX.commits.sha[state.point])}`);
  // Legend-hidden series, only those the chips currently select.
  for (const params of g.series) {
    const key = seriesKey(params);
    if (state.hidden.has(key) && params.every((v, n) => state.selected[n].has(v))) {
      parts.push(`hide=${encodeURIComponent(key)}`);
    }
  }
  const hash = "#" + g.name + (parts.length ? "?" + parts.join("&") : "");
  history.replaceState(null, "", hash);
}

// Options present in the URL set the toggles; absent ones keep the current
// state, so following a tree link keeps e.g. log scale on.
function applyViewOptions(opts, hidden = []) {
  state.hidden = new Set(hidden);
  const set = (id, key, on) => { if (opts.has(key)) $(id).checked = opts.get(key) === on; };
  set("opt-log", "log", "1");
  set("opt-order", "x", "commit");
  if (opts.has("env")) $("opt-env").checked = opts.get("env") !== "0";
  set("opt-env-all", "envall", "1");
  const from = commitBySha(opts.get("from"));
  const to = commitBySha(opts.get("to"));
  state.zoom = from !== null && to !== null && from < to ? { from, to } : null;
  state.point = commitBySha(opts.get("at"));
}

// ---- sidebar tree ----------------------------------------------------------

function treeParts(name) {
  const p = name.split(".");
  if (p.length >= 3) return [p.slice(0, -2).join("."), p[p.length - 2], p[p.length - 1]];
  return [p[0], null, p.slice(1).join(".")];
}

function buildTree() {
  const modules = new Map();
  for (const name of Object.keys(INDEX.benchmarks)) {
    const [mod, cls, leaf] = treeParts(name);
    if (!modules.has(mod)) modules.set(mod, new Map());
    const classes = modules.get(mod);
    if (!classes.has(cls)) classes.set(cls, []);
    classes.get(cls).push([name, leaf]);
  }
  // Search matches the name and the parameter values, so "groupby cumsum"
  // finds benchmarks with a 'cumsum' parameter.
  const searchText = (name) =>
    [name, ...INDEX.benchmarks[name].param_values.flat()].join(" ").toLowerCase();
  const badge = () => el("span", { class: "badge", title: `Regression in the last ${RECENT_DAYS} days` }, "▲");
  const link = ([name, leaf]) =>
    el("li", { "data-search": searchText(name) },
      el("a", { href: "#" + name, "data-full": name }, leaf, recentRegressed.has(name) ? badge() : ""));
  const summary = (label, count, regressed) =>
    el("summary", {}, label, " ", el("span", { class: "count" }, String(count)),
      regressed ? el("span", { class: "badge", title: `${regressed} with a regression in the last ${RECENT_DAYS} days` }, ` ▲${regressed}`) : "");
  const nRegressed = (items) => items.filter(([name]) => recentRegressed.has(name)).length;
  const tree = $("tree");
  for (const [mod, classes] of modules) {
    const all = [...classes.values()].flat();
    const modNode = el("details", {}, summary(mod, all.length, nRegressed(all)));
    for (const [cls, items] of classes) {
      if (cls === null) {
        modNode.append(el("ul", {}, ...items.map(link)));
      } else {
        modNode.append(el("details", {}, summary(cls, items.length, nRegressed(items)), el("ul", {}, ...items.map(link))));
      }
    }
    tree.append(modNode);
  }
}

function shortName(name) {
  return name.split(".").slice(-2).join(".");
}

function filterTree() {
  const q = $("search").value.trim().toLowerCase();
  // Every word must appear somewhere in the name, in any order.
  const words = q.split(/\s+/).filter(Boolean);
  for (const li of $("tree").querySelectorAll("li")) {
    li.hidden = words.length > 0 && !words.every((w) => li.dataset.search.includes(w));
  }
  renderCommitSearch(q);
  // Innermost first, so a module sees its classes' final visibility.
  const groups = [...$("tree").querySelectorAll("details")].reverse();
  for (const d of groups) {
    const any = [...d.querySelectorAll("li")].some((li) => !li.hidden);
    d.hidden = !any;
    if (q) d.open = any;
  }
}

// A query that looks like a commit SHA or a PR number ("#12345" or "12345")
// also lists matching benchmarked commits.
function renderCommitSearch(q) {
  const box = $("search-commit");
  const pr = /^#?(\d{3,6})$/.exec(q);
  const isSha = /^[0-9a-f]{7,40}$/.test(q);
  if (!pr && !isSha) {
    box.replaceChildren();
    return;
  }
  const hits = [];
  if (pr) {
    INDEX.commits.pr?.forEach((n, i) => { if (n === Number(pr[1])) hits.push(i); });
  }
  if (isSha) {
    INDEX.commits.sha.forEach((sha, i) => { if (sha.startsWith(q)) hits.push(i); });
  }
  if (!hits.length) {
    box.replaceChildren(el("div", { class: "muted" },
      pr && !isSha ? `No benchmarked commit from PR #${pr[1]}.` : "No benchmarked commit starts with ",
      pr && !isSha ? "" : el("code", {}, q), pr && !isSha ? "" : "."));
    return;
  }
  box.replaceChildren(...hits.slice(0, 5).map((i) => {
    const sha = INDEX.commits.sha[i];
    const env = envByCommit.get(i);
    return el("div", { class: "commit-hit" },
      el("a", { href: COMMIT_URL + sha, target: "_blank" }, el("code", {}, shortSha(sha))),
      " ", titleNode(i), el("br"),
      ` committed ${fmtDate(INDEX.commits.date[i])} UTC — `,
      el("a", { href: `#commit/${sha}` }, detCountText(i)),
      env && envIsKey(env) ? el("div", { class: "ev-env" }, "Environment changed: " + envSummary(env)) : "",
    );
  }));
}

function markActive(name) {
  for (const a of $("tree").querySelectorAll("a.active")) a.classList.remove("active");
  const a = $("tree").querySelector(`a[data-full="${CSS.escape(name)}"]`);
  if (!a) return;
  a.classList.add("active");
  for (let d = a.closest("details"); d; d = d.parentElement.closest("details")) d.open = true;
  a.scrollIntoView({ block: "nearest" });
}

// ---- environment changes ---------------------------------------------------

function envIsKey(e) {
  return Boolean(e.format_change) || e.n_key > 0;
}

function envShown(e, allBoxId) {
  return envIsKey(e) || $(allBoxId).checked;
}

function changesNode(e) {
  if (e.format_change) {
    return el("div", { class: "changes" }, `environment tool changed: ${e.format_change[0]} → ${e.format_change[1]}`);
  }
  const line = ([p, a, b]) => el("div", {}, `${p}: ${a ?? "—"} → ${b ?? "—"}`);
  const key = e.changes.slice(0, e.n_key);
  const rest = e.changes.slice(e.n_key);
  const node = el("div", { class: "changes" }, ...key.map(line));
  if (rest.length) {
    node.append(el("details", {}, el("summary", {}, `${key.length ? "show all" : "show"} ${rest.length} other packages`), ...rest.map(line)));
  }
  return node;
}

function envSummary(e) {
  if (e.format_change) return "environment tool changed";
  const key = e.changes.slice(0, e.n_key).map(([p, , b]) => `${p} ${b ?? "removed"}`);
  const n = e.changes.length;
  return (key.length ? key.join(", ") + "; " : "") + `${n} package${n === 1 ? "" : "s"} changed`;
}

// ---- home ------------------------------------------------------------------

// ---- detections ------------------------------------------------------------

function indexDetections() {
  const d = INDEX.detections;
  const cutoff = INDEX.commits.date[INDEX.commits.date.length - 1] - RECENT_DAYS * 864e5;
  for (let k = 0; k < d.i.length; k++) {
    const det = { name: d.name[k], params: d.params[k], kind: d.kind[k], step: d.step[k] };
    if (!detByCommit.has(d.i[k])) detByCommit.set(d.i[k], []);
    detByCommit.get(d.i[k]).push(det);
    if (det.kind === 1 && INDEX.commits.date[d.i[k]] >= cutoff) recentRegressed.add(det.name);
  }
}

function detCountText(i) {
  const dets = detByCommit.get(i) || [];
  const r = dets.filter((d) => d.kind === 1).length;
  const imp = dets.length - r;
  if (!dets.length) return "no changes detected";
  return [r && `${r} regression${r === 1 ? "" : "s"}`, imp && `${imp} improvement${imp === 1 ? "" : "s"}`].filter(Boolean).join(", ");
}

// Same fragment format as the links in regression issues.
function benchHref(name, params) {
  const names = INDEX.benchmarks[name]?.param_names || [];
  const parts = params.map((v, n) => `p-${encodeURIComponent(names[n] ?? `param${n + 1}`)}=${encodeURIComponent(v)}`);
  return "#" + name + (parts.length ? "?" + parts.join("&") : "");
}

function paramsLabel(name, params) {
  const names = INDEX.benchmarks[name]?.param_names || [];
  return params.map((v, n) => `${names[n] ?? `param${n + 1}`}=${v}`).join(", ");
}

function fmtStep(step) {
  if (step == null) return "";
  const pct = step * 100;
  return (pct > 0 ? "+" : "") + pct.toFixed(Math.abs(pct) < 10 ? 1 : 0) + "%";
}

function compareLink(i, text = "changes since previous benchmarked commit") {
  if (i === 0) return "";
  return el("a", { href: COMPARE_URL + INDEX.commits.sha[i - 1] + "..." + INDEX.commits.sha[i], target: "_blank" }, text);
}

// A list of detections, largest change first, the rest behind "show more".
function detectionList(dets) {
  const sorted = [...dets].sort((a, b) => Math.abs(b.step ?? 0) - Math.abs(a.step ?? 0));
  const item = (d) =>
    el("li", {},
      el("a", { href: benchHref(d.name, d.params) }, d.name),
      d.params.length ? el("span", { class: "muted" }, ` ${paramsLabel(d.name, d.params)}`) : "",
      " ", el("span", { class: d.kind === 1 ? "step-bad" : "step-good" }, fmtStep(d.step)),
    );
  const list = el("ul", { class: "det-list" }, ...sorted.slice(0, GROUP_LIMIT).map(item));
  if (sorted.length > GROUP_LIMIT) {
    list.append(el("li", {}, el("details", {}, el("summary", {}, `${sorted.length - GROUP_LIMIT} more`), el("ul", {}, ...sorted.slice(GROUP_LIMIT).map(item)))));
  }
  return list;
}

// One block per commit: date, links, environment note, detections.
function commitGroup(i, dets) {
  const env = envByCommit.get(i);
  const brk = INDEX.breaks.find((b) => b.i === i);
  return el("div", { class: "det-group" },
    el("div", {},
      el("strong", {}, fmtDate(INDEX.commits.date[i]).slice(0, 10)), " ", commitLink(i), " ", titleNode(i), " — ",
      el("a", { href: `#commit/${INDEX.commits.sha[i]}` }, `${dets.length} benchmark${dets.length === 1 ? "" : "s"}`),
      " — ", compareLink(i),
    ),
    env && envIsKey(env) ? el("div", { class: "ev-env" }, "Environment changed at this commit: " + envSummary(env)) : "",
    brk ? el("div", { class: "ev-break" }, "Measurement changed at this commit: " + brk.reason) : "",
    detectionList(dets),
  );
}

function renderDetectionsHome() {
  const cutoff = INDEX.commits.date[INDEX.commits.date.length - 1] - RECENT_DAYS * 864e5;
  const groups = (kind) =>
    [...detByCommit.entries()]
      .filter(([i]) => INDEX.commits.date[i] >= cutoff)
      .map(([i, dets]) => [i, dets.filter((d) => d.kind === kind)])
      .filter(([, dets]) => dets.length)
      .sort((a, b) => b[0] - a[0]);
  const reg = groups(1);
  $("recent-regressions").replaceChildren(
    ...(reg.length ? reg.map(([i, dets]) => commitGroup(i, dets)) : [el("p", {}, `None in the last ${RECENT_DAYS} days.`)]),
  );
  const imp = groups(-1);
  const box = $("recent-improvements");
  box.querySelector("summary h3").textContent = `Recent improvements (${imp.reduce((a, [, d]) => a + d.length, 0)})`;
  box.querySelector("div").replaceChildren(
    ...(imp.length ? imp.map(([i, dets]) => commitGroup(i, dets)) : [el("p", {}, `None in the last ${RECENT_DAYS} days.`)]),
  );
}

function showSection(id) {
  for (const s of ["home", "bench", "commit"]) $(s).hidden = s !== id;
}

function showCommit(shaPrefix) {
  const i = INDEX.commits.sha.findIndex((sha) => sha.startsWith(shaPrefix));
  showSection("commit");
  if (i < 0) {
    $("commit-title").textContent = `Commit ${shaPrefix} was not benchmarked`;
    for (const id of ["commit-info", "commit-env", "commit-regressions", "commit-improvements"]) $(id).replaceChildren();
    return;
  }
  const sha = INDEX.commits.sha[i];
  $("commit-title").replaceChildren(commitTitle(i) ?? "Commit", " ", el("code", {}, shortSha(sha)));
  const pr = prLink(i);
  $("commit-info").replaceChildren(
    pr ? el("span", {}, "Pull request ", pr, ". ") : "",
    `Committed ${fmtDate(INDEX.commits.date[i])} UTC. `,
    el("a", { href: COMMIT_URL + sha, target: "_blank" }, "Commit on GitHub"), " — ", compareLink(i),
  );
  const env = envByCommit.get(i);
  const brk = INDEX.breaks.find((b) => b.i === i);
  $("commit-env").replaceChildren(
    brk ? el("p", { class: "ev-break" }, "Measurement changed at this commit; results before and after are not compared: " + brk.reason) : "",
    env ? el("div", { class: "ev-env" }, "Environment changed at this commit: " + envSummary(env), changesNode(env)) : "",
  );
  const dets = detByCommit.get(i) || [];
  const recent = i > INDEX.commits.sha.length - 11;
  const none = el("p", { class: "muted" }, recent ? "None detected yet: changes are detected once 10 later commits have been benchmarked." : "None.");
  const reg = dets.filter((d) => d.kind === 1);
  const imp = dets.filter((d) => d.kind === -1);
  $("commit-regressions").replaceChildren(reg.length ? detectionList(reg) : none);
  $("commit-improvements").replaceChildren(imp.length ? detectionList(imp) : none.cloneNode(true));
}

function ago(ms) {
  const h = (Date.now() - ms) / 3.6e6;
  if (h < 1) return "less than an hour ago";
  if (h < 48) return `${Math.round(h)} hours ago`;
  return `${Math.round(h / 24)} days ago`;
}

function commitLink(i) {
  const sha = INDEX.commits.sha[i];
  return el("a", { href: COMMIT_URL + sha, target: "_blank" }, el("code", {}, shortSha(sha)));
}

function renderHome() {
  const st = INDEX.status;
  const n = INDEX.commits.sha.length;
  const status = ["Latest benchmarked commit ", commitLink(st.last_commit), `, committed ${fmtDate(INDEX.commits.date[st.last_commit])} UTC`];
  if (st.last_run) status.push(`; benchmarked ${fmtDate(st.last_run)} UTC (${ago(st.last_run)})`);
  status.push(".");
  $("status").replaceChildren(...status);

  const first = fmtDate(INDEX.commits.date[0]).slice(0, 10);
  const last = fmtDate(INDEX.commits.date[n - 1]).slice(0, 10);
  $("summary").textContent = `${Object.keys(INDEX.benchmarks).length} benchmarks over ${n} commits, ${first} to ${last}.`;

  const failing = $("failing");
  failing.hidden = st.failing.length === 0;
  failing.querySelector("ul").replaceChildren(...st.failing.map(({ name, since }) => {
    const runs = st.last_commit - since + 1;
    return el("li", {},
      el("a", { href: "#" + name }, name),
      ` failing since `, commitLink(since),
      ` (${fmtDate(INDEX.commits.date[since]).slice(0, 10)}, ${runs} benchmarked commit${runs === 1 ? "" : "s"})`,
    );
  }));
  renderHomeEnvs();
  renderDetectionsHome();
}

function renderHomeEnvs() {
  const shown = INDEX.env_changes.filter((e) => envShown(e, "home-env-all")).slice(-15).reverse();
  $("recent-envs").replaceChildren(...shown.map((e) =>
    el("li", {},
      `${fmtDate(INDEX.commits.date[e.i]).slice(0, 10)} `,
      commitLink(e.i),
      ` ${envSummary(e)}`,
      changesNode(e),
    ),
  ));
}

// ---- recently viewed (per browser) -----------------------------------------

const RECENT_KEY = "asv-runner-recent";

function loadRecent() {
  try {
    const r = JSON.parse(localStorage.getItem(RECENT_KEY));
    return Array.isArray(r) ? r.filter((n) => INDEX.benchmarks[n]) : [];
  } catch {
    return [];
  }
}

function pushRecent(name) {
  const r = [name, ...loadRecent().filter((n) => n !== name)].slice(0, 8);
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(r));
  } catch {
    // Storage unavailable (private window, blocked); the list just isn't kept.
  }
  renderRecent(r);
}

function renderRecent(r = loadRecent()) {
  const box = $("recent");
  box.hidden = r.length === 0;
  box.querySelector("ul").replaceChildren(...r.map((n) => el("li", {}, el("a", { href: "#" + n, title: n }, shortName(n)))));
}

// ---- benchmark view --------------------------------------------------------

function paramValues(g, n) {
  return INDEX.benchmarks[g.name].param_values[n];
}

async function loadGraph(name) {
  if (!graphCache.has(name)) {
    const resp = await fetch("data/graphs/" + encodeURIComponent(name) + ".json");
    if (!resp.ok) throw new Error(`No data for ${name}`);
    const g = await resp.json();
    g.pos = new Map(g.commits.map((i, idx) => [i, idx]));
    graphCache.set(name, g);
  }
  return graphCache.get(name);
}

async function showBench(name, params, opts = new Map(), hidden = []) {
  if (!INDEX.benchmarks[name]) {
    showSection("home");
    $("summary").textContent = `Unknown benchmark: ${name}`;
    return;
  }
  const g = await loadGraph(name);
  state.name = name;
  state.graph = g;
  state.selected = g.param_names.map((p, n) => {
    const all = paramValues(g, n);
    const want = params.get(p);
    const sel = want ? all.filter((v) => want.has(v)) : all;
    return new Set(sel.length ? sel : all);
  });
  applyViewOptions(opts, hidden);
  showSection("bench");
  $("bench-name").textContent = name;
  $("point").replaceChildren();
  markActive(name);
  pushRecent(name);
  renderChips();
  renderEvents();
  await drawChart();
  const idx = state.point === null ? undefined : g.pos.get(state.point);
  if (idx === undefined) state.point = null;
  else showPoint(idx);
  writeHash();
}

function renderChips() {
  const g = state.graph;
  const rows = g.param_names.map((p, n) => {
    const sel = state.selected[n];
    const all = paramValues(g, n);
    const update = () => { writeHash(); renderChips(); drawChart(); };
    const allBtn = el("button", { class: "all", onclick: () => { all.forEach((v) => sel.add(v)); update(); } }, "all");
    const chips = all.map((v) =>
      el("button", {
        class: sel.has(v) ? "on" : "",
        title: "Click to toggle, shift-click to show only this value",
        onclick: (ev) => {
          if (ev.shiftKey) { sel.clear(); sel.add(v); }
          else if (sel.has(v)) sel.delete(v);
          else sel.add(v);
          update();
        },
      }, v),
    );
    return el("div", { class: "chip-row" }, el("span", { class: "label" }, p), allBtn, ...chips);
  });
  $("chips").replaceChildren(...rows);
}

function xValue(i) {
  return $("opt-order").checked ? i : INDEX.commits.date[i] / 1000;
}

function seriesLabel(g, k) {
  if (!g.param_names.length) return "result";
  return g.param_names.map((p, n) => `${p}=${g.series[k][n]}`).join(", ");
}

// Benchmarks with many series keep their values in chunk files of
// g.chunk_size series; fetch only the chunks holding the series shown.
async function loadChunks(g, shown) {
  if (g.values) return;
  g.chunkData ??= {};
  const need = [...new Set(shown.map((k) => Math.floor(k / g.chunk_size)))].filter((c) => !(c in g.chunkData));
  await Promise.all(need.map(async (c) => {
    const resp = await fetch(`data/graphs/${encodeURIComponent(g.name)}.chunk${c}.json`);
    if (!resp.ok) throw new Error(`No data for ${g.name} chunk ${c}`);
    g.chunkData[c] = await resp.json();
  }));
}

function seriesValues(g, k) {
  if (g.values) return g.values[k];
  return g.chunkData[Math.floor(k / g.chunk_size)][k % g.chunk_size];
}

function color(k, n) {
  const hue = Math.round((360 * k) / Math.max(n, 1) + 200) % 360;
  return `hsl(${hue} 65% 50%)`;
}

// Shows the commit under the cursor: date, SHA, title, PR, and any detection.
function tooltipPlugin() {
  let tip;
  const init = (u) => {
    tip = el("div", { class: "tooltip", hidden: "" });
    u.over.append(tip);
    u.over.addEventListener("mouseleave", () => { tip.hidden = true; });
  };
  const setCursor = (u) => {
    const { idx, left, top } = u.cursor;
    if (idx == null || left < 0) {
      tip.hidden = true;
      return;
    }
    const g = state.graph;
    const i = g.commits[idx];
    const sha = INDEX.commits.sha[i];
    const pr = INDEX.commits.pr?.[i];
    const lines = [
      el("div", { class: "muted" }, `${fmtDate(INDEX.commits.date[i])} UTC · ${shortSha(sha)}${pr ? ` · #${pr}` : ""}`),
    ];
    if (commitTitle(i)) lines.push(el("div", { class: "tooltip-title" }, commitTitle(i)));
    for (const [di, sid, kind, step] of g.detections) {
      if (di === i && state.shownIndex.has(sid)) {
        lines.push(el("div", { class: kind === 1 ? "step-bad" : "step-good" }, `${kind === 1 ? "Regression" : "Improvement"} ${fmtStep(step)}`));
      }
    }
    tip.replaceChildren(...lines);
    tip.hidden = false;
    // Keep the tooltip inside the plot: flip to the left of the cursor near
    // the right edge.
    const w = tip.offsetWidth;
    const x = left + 14 + w > u.over.clientWidth ? left - 14 - w : left + 14;
    tip.style.left = `${Math.max(0, x)}px`;
    tip.style.top = `${Math.max(0, top - 10)}px`;
  };
  return { hooks: { init: [init], setCursor: [setCursor] } };
}

function markerPlugin() {
  const draw = (u) => {
    const g = state.graph;
    const ctx = u.ctx;
    const { left, top, width, height } = u.bbox;
    const inRange = (x) => x >= left && x <= left + width;
    ctx.save();
    ctx.lineWidth = devicePixelRatio;
    ctx.strokeStyle = cssVar("--version");
    ctx.setLineDash([5 * devicePixelRatio, 4 * devicePixelRatio]);
    const vline = (i) => {
      const x = Math.round(u.valToPos(xValue(i), "x", true));
      if (!inRange(x)) return;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, top + height);
      ctx.stroke();
    };
    for (const i of g.version_changes) vline(i);
    // Break points (breaks.toml): measurement changed; solid and thicker.
    ctx.setLineDash([]);
    ctx.lineWidth = 2 * devicePixelRatio;
    ctx.strokeStyle = cssVar("--break");
    for (const b of INDEX.breaks) vline(b.i);
    ctx.lineWidth = devicePixelRatio;
    // Detections on the shown series: ▲ regression above the point, ▼
    // improvement below it.
    const sz = 5 * devicePixelRatio;
    for (const [i, sid, kind] of g.detections) {
      const n = state.shownIndex.get(sid);
      const idx = g.pos.get(i);
      if (n === undefined || idx === undefined) continue;
      const y = u.data[n + 1][idx];
      if (y == null) continue;
      const x = u.valToPos(xValue(i), "x", true);
      const py = u.valToPos(y, "y", true);
      if (!inRange(x)) continue;
      ctx.fillStyle = cssVar(kind === 1 ? "--bad" : "--good");
      const d = kind === 1 ? -1 : 1; // canvas y grows downward
      ctx.beginPath();
      ctx.moveTo(x, py + d * sz * 0.6);
      ctx.lineTo(x - sz, py + d * sz * 2.4);
      ctx.lineTo(x + sz, py + d * sz * 2.4);
      ctx.closePath();
      ctx.fill();
    }
    if ($("opt-env").checked) {
      ctx.setLineDash([]);
      ctx.fillStyle = cssVar("--env");
      const s = 6 * devicePixelRatio;
      const first = g.commits[0];
      const last = g.commits[g.commits.length - 1];
      for (const e of INDEX.env_changes) {
        if (e.i < first || e.i > last || !envShown(e, "opt-env-all")) continue;
        const x = u.valToPos(xValue(e.i), "x", true);
        if (!inRange(x)) continue;
        const y = top + height;
        ctx.beginPath();
        ctx.moveTo(x, y - s * 1.6);
        ctx.lineTo(x - s, y);
        ctx.lineTo(x + s, y);
        ctx.closePath();
        ctx.fill();
      }
    }
    ctx.restore();
  };
  return { hooks: { draw: [draw] } };
}

let drawId = 0;
let restoringZoom = false; // setScale calls made while restoring a zoom

async function drawChart() {
  const g = state.graph;
  if (!g) return;
  const id = ++drawId;
  const order = $("opt-order").checked;
  const unit = INDEX.benchmarks[g.name].unit;
  const visible = [];
  g.series.forEach((params, k) => {
    if (params.every((v, n) => state.selected[n].has(v))) visible.push(k);
  });
  const shown = visible.slice(0, MAX_SERIES);
  await loadChunks(g, shown);
  if (id !== drawId) return; // a newer draw started while fetching
  state.shownIndex = new Map(shown.map((k, n) => [k, n]));
  $("series-note").textContent =
    visible.length > MAX_SERIES ? `Showing ${MAX_SERIES} of ${visible.length} series; narrow the parameters to see the rest.` :
    visible.length === 0 ? "No series selected." : "";

  const xs = g.commits.map(xValue);
  const data = [xs, ...shown.map((k) => seriesValues(g, k))];
  const fg = cssVar("--muted");
  const grid = { stroke: cssVar("--border"), width: 1 };
  const opts = {
    width: $("chart").clientWidth,
    height: CHART_HEIGHT,
    plugins: [markerPlugin(), tooltipPlugin()],
    scales: { x: { time: !order }, y: { distr: $("opt-log").checked ? 3 : 1 } },
    cursor: {
      focus: { prox: 16 },
      // The point under the cursor: filled with the series color, ringed with
      // the page background so it stands out from the line.
      points: {
        size: 12,
        width: 2.5,
        fill: (u, si) => u.series[si]._stroke,
        stroke: () => cssVar("--bg"),
      },
    },
    focus: { alpha: 0.3 },
    legend: { live: true },
    axes: [
      { stroke: fg, grid, ticks: grid },
      { stroke: fg, grid, ticks: grid, size: 80, values: (u, vals) => vals.map((v) => fmtVal(v, unit)) },
    ],
    series: [
      {
        label: order ? "commit" : "date",
        value: (u, v, si, idx) => {
          if (idx == null) return "";
          const i = g.commits[idx];
          return `${fmtDate(INDEX.commits.date[i])} ${shortSha(INDEX.commits.sha[i])}`;
        },
      },
      ...shown.map((k, n) => ({
        show: !state.hidden.has(seriesKey(g.series[k])),
        label: seriesLabel(g, k),
        stroke: color(n, shown.length),
        width: 1.25,
        spanGaps: true,
        points: { show: g.commits.length < 150 },
        value: (u, v) => fmtVal(v, unit),
      })),
    ],
  };
  // Record drag-zoom and its double-click reset as commit positions, and
  // series hidden or shown from the legend.
  opts.hooks = {
    setSeries: [
      (u, si, change) => {
        if (si == null || !change || !("show" in change)) return; // focus changes
        const key = seriesKey(g.series[shown[si - 1]]);
        if (change.show) state.hidden.delete(key);
        else state.hidden.add(key);
        writeHash();
      },
    ],
    setScale: [
      (u, key) => {
        if (key !== "x" || restoringZoom) return;
        const { min, max } = u.scales.x;
        const lo = xs.findIndex((x) => x >= min);
        const hi = xs.findLastIndex((x) => x <= max);
        const full = lo <= 0 && hi >= xs.length - 1;
        state.zoom = full || lo < 0 || hi <= lo ? null : { from: g.commits[lo], to: g.commits[hi] };
        writeHash();
      },
    ],
  };
  if (state.plot) state.plot.destroy();
  restoringZoom = true;
  state.plot = new uPlot(opts, data, $("chart"));
  if (state.zoom) state.plot.setScale("x", { min: xValue(state.zoom.from), max: xValue(state.zoom.to) });
  restoringZoom = false;
  // A drag-to-zoom also ends in a click; only treat it as a click on a point
  // when the mouse did not move.
  let downX = null;
  state.plot.over.addEventListener("mousedown", (ev) => { downX = ev.clientX; });
  // A single click opens the commit, but only once it is clear the click is
  // not the first half of a double-click (which resets the zoom): opening a
  // tab right away moves focus there and the second click never arrives.
  let clickTimer = null;
  state.plot.over.addEventListener("click", (ev) => {
    if (downX !== null && Math.abs(ev.clientX - downX) > 3) return;
    clearTimeout(clickTimer);
    if (ev.detail > 1) return; // second click of a double-click
    const idx = state.plot.cursor.idx;
    if (idx == null) return;
    clickTimer = setTimeout(() => {
      showPoint(idx);
      const sha = INDEX.commits.sha[state.graph.commits[idx]];
      window.open(COMMIT_URL + sha, "_blank", "noopener");
    }, DOUBLE_CLICK_MS);
  });
}

function showPoint(idx) {
  const g = state.graph;
  const i = g.commits[idx];
  state.point = i;
  writeHash();
  const sha = INDEX.commits.sha[i];
  const node = el("div", {},
    el("strong", {}, fmtDate(INDEX.commits.date[i])), " — commit ",
    el("a", { href: COMMIT_URL + sha, target: "_blank" }, el("code", {}, shortSha(sha))),
    " ", titleNode(i),
  );
  if (idx > 0) {
    const prev = INDEX.commits.sha[g.commits[idx - 1]];
    node.append(" — ", el("a", { href: COMPARE_URL + prev + "..." + sha, target: "_blank" }, "changes since previous benchmarked commit"));
  }
  if (g.version_changes.includes(i)) {
    node.append(el("div", { class: "ev-version" }, "Benchmark code changed at this commit."));
  }
  const brk = INDEX.breaks.find((b) => b.i === i);
  if (brk) {
    node.append(el("div", { class: "ev-break" }, "Measurement changed at this commit; results before and after are not compared: " + brk.reason));
  }
  const here = g.detections.filter(([di]) => di === i);
  for (const [, sid, kind, step] of here) {
    node.append(el("div", { class: kind === 1 ? "step-bad" : "step-good" },
      `${kind === 1 ? "Regression" : "Improvement"} ${fmtStep(step)}: ${seriesLabel(g, sid)}`));
  }
  const env = envByCommit.get(i);
  if (env) node.append(el("div", { class: "ev-env" }, "Environment changed at this commit: " + envSummary(env)), changesNode(env));
  node.append(el("div", {}, el("a", { href: `#commit/${sha}` }, `All benchmarks at this commit: ${detCountText(i)}`)));
  $("point").replaceChildren(node);
}

function renderEvents() {
  const g = state.graph;
  const pos = new Map(g.commits.map((i, idx) => [i, idx]));
  const events = [];
  for (const i of g.version_changes) events.push({ i, kind: "version" });
  for (const b of INDEX.breaks) if (pos.has(b.i)) events.push({ i: b.i, kind: "break", b });
  for (const e of INDEX.env_changes) {
    if (pos.has(e.i) && envShown(e, "opt-env-all")) events.push({ i: e.i, kind: "env", e });
  }
  events.sort((a, b) => b.i - a.i);
  const label = (kind, e, b) =>
    kind === "version" ? el("span", { class: "ev-version" }, "benchmark code changed")
    : kind === "break" ? el("span", { class: "ev-break" }, "measurement changed: " + b.reason)
    : el("span", { class: "ev-env" }, envSummary(e));
  const items = events.map(({ i, kind, e, b }) =>
    el("li", {},
      el("a", { href: "javascript:void 0", onclick: () => showPoint(pos.get(i)) }, fmtDate(INDEX.commits.date[i]).slice(0, 10)),
      " ",
      label(kind, e, b),
      commitTitle(i) ? el("span", { class: "muted" }, ` — ${commitTitle(i)}`) : "",
    ),
  );
  $("events").replaceChildren(...(items.length ? items : [el("li", {}, "None.")]));
}

// ---- routing ---------------------------------------------------------------

function route() {
  const parsed = parseHash();
  if (!parsed || !parsed.name) {
    state.graph = null;
    showSection("home");
    return;
  }
  if (parsed.name.startsWith("commit/")) {
    state.graph = null;
    showCommit(parsed.name.slice("commit/".length).toLowerCase());
    return;
  }
  showBench(parsed.name, parsed.params, parsed.opts, parsed.hidden).catch((err) => {
    showSection("home");
    $("summary").textContent = String(err);
  });
}

async function main() {
  INDEX = await (await fetch("data/index.json")).json();
  for (const e of INDEX.env_changes) envByCommit.set(e.i, e);
  indexDetections();
  buildTree();
  renderHome();
  renderRecent();
  $("search").addEventListener("input", filterTree);
  $("search").addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { $("search").value = ""; filterTree(); $("search").blur(); }
  });
  document.addEventListener("keydown", (ev) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if (ev.key === "/" && !typing && !ev.ctrlKey && !ev.metaKey) {
      ev.preventDefault();
      $("search").focus();
      $("search").select();
    }
  });
  for (const id of ["opt-log", "opt-order", "opt-env"]) {
    $(id).addEventListener("change", () => { writeHash(); drawChart(); });
  }
  $("opt-env-all").addEventListener("change", () => { writeHash(); drawChart(); renderEvents(); });
  $("copy-link").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(location.href);
      $("copy-link").textContent = "Copied";
    } catch {
      $("copy-link").textContent = "Copy failed";
    }
    setTimeout(() => { $("copy-link").textContent = "Copy link"; }, 1500);
  });
  $("home-env-all").addEventListener("change", renderHomeEnvs);
  window.addEventListener("hashchange", route);
  window.addEventListener("resize", () => {
    if (state.plot) state.plot.setSize({ width: $("chart").clientWidth, height: CHART_HEIGHT });
  });
  route();
}

main();
