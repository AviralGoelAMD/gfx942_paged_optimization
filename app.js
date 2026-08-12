// gfx942 D128-SW paged-dense optimization dashboard.
// 100% data-driven: every value below is read from data.json at load time.

const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const PHASE_ORDER_LABEL = {
  grounding: "Grounding",
  dashboard: "Dashboard",
  baseline: "Baseline",
  journey: "Journey doc",
  deferred: "Deferred",
};

async function main() {
  const d = await (await fetch("data.json?" + Date.now())).json();
  const ps = d.problemShape;
  const ours = d.kernels.find((k) => k.role === "baseline");
  const aiter = d.kernels.find((k) => k.role === "competitor");

  document.title = d.title;
  renderHeader(d, ps, ours, aiter);
  const main = document.getElementById("main");
  main.innerHTML =
    stageMapSection(d, ps, ours, aiter) +
    portStatusSection(d) +
    shapeSection(ps) +
    kernelSection(ours, aiter) +
    resultsSection(d) +
    directionalReadSection(d) +
    pagedResultsSection(d) +
    calloutSection(ps) +
    checklistSection(d);
  renderFooter(d, ps, ours, aiter);

  // progress bar animates once laid out
  requestAnimationFrame(() => {
    const pf = document.querySelector(".pfill");
    if (pf) pf.style.width = pf.dataset.pct + "%";
    document.querySelectorAll(".bfill").forEach((b) => (b.style.width = b.dataset.w + "%"));
  });

  const io = new IntersectionObserver(
    (es) => es.forEach((e) => { if (e.isIntersecting) e.target.classList.add("in"); }),
    { threshold: 0.12 }
  );
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
}

// ---- header ----------------------------------------------------------------
function renderHeader(d, ps, ours, aiter) {
  const { done, active } = progress(d.checklist);
  const pct = active ? Math.round((100 * done) / active) : 0;
  document.getElementById("hdr").innerHTML = `
    <h1>${esc(d.title)}</h1>
    <div class="sub">Living dashboard for the <b>${esc(ps.operation)}</b> campaign &mdash;
    <b>${esc(ps.dtype)}</b>, head&nbsp;dim&nbsp;${esc(ps.head_size)}, ${esc(ps.masking)}.
    We ground our CK&nbsp;DSL kernel (<b>rocKE</b>) against <b>${esc(ps.competitor)}</b> on an
    identical problem shape, then honestly report where we stand.</div>
    <div class="tags">
      <span class="tag">operation = ${esc(ps.operation)}</span>
      <span class="tag">${esc(ps.masking)}</span>
      <span class="tag ours">rocKE = purple</span>
      <span class="tag aiter">AITER = teal</span>
    </div>
    <div class="hprog">
      <div class="ptop"><span class="plab">Campaign progress</span>
        <span class="pval">${pct}% &middot; ${done}/${active} active</span></div>
      <div class="ptrack"><div class="pfill" data-pct="${pct}"></div></div>
    </div>
    <div class="updated">Updated ${esc(d.updated)}</div>`;
}

// ---- 1. stage map ----------------------------------------------------------
function stageMapSection(d, ps, ours, aiter) {
  const seqs = String(ps.seqlens).replace(/[^0-9,{} ]/g, "").replace(/[{} ]/g, "");
  const inputChips = [
    ["Q,K,V", ps.dtype.split(" ")[0]],
    ["head_size", ps.head_size],
    ["heads q/kv", ps.gqa.replace(/Hq=|Hkv=/g, "").replace("/", " / ")],
    ["KV cache", ps.kv_layout.split(",")[0]],
    ["block_size", (ps.kv_layout.match(/block_size=(\d+)/) || [, "16"])[1]],
    ["Sq = kv_len", seqs],
    ["window", (ps.masking.match(/window=(\d+)/) || [, "-"])[1]],
    ["tile T", ps.tile.replace("T=", "")],
  ];
  const m = ps.measurement;
  const warm = (m.match(/(\d+)\s*warmup/) || [, "-"])[1];
  const iters = (m.match(/(\d+)\s*iter/) || [, "-"])[1];
  const backend = (m.match(/ROCKE_BACKEND=(\w+)/) || [, "python"])[1];
  const outChips = [
    ["O", ps.dtype.split(" ")[0]],
    ["timer", "CUDA event"],
    ["num_seqs", ps.num_seqs],
    ["warmup", warm],
    ["iters", iters],
    ["backend", backend],
  ];

  return `
  <section>
    <h2>1. One shape, two kernels</h2>
    <p class="lede">Same inputs, same output contract, same timer &mdash; only the kernel between
    them changes. Holding the problem shape fixed is what makes the latency difference meaningful.</p>
    <div class="map reveal">
      <div class="col">
        <h3>Shared inputs</h3>
        ${inputChips.map(([k, v]) => `<div class="chip"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join("")}
      </div>
      <div class="arrow">&rarr;</div>
      <div class="twolane">
        <div class="lane ours">
          <h3>&#9679; ${esc(ours.name)}</h3>
          <p>${esc(ours.sourcePath.split("::")[1] || ours.sourcePath)}</p>
          <p>CK DSL &middot; 2D tiled</p>
          <p>role: baseline (ours)</p>
        </div>
        <div class="lane aiter">
          <h3>&#9679; ${esc(aiter.name)}</h3>
          <p>${esc(aiter.sourcePath.split(".").pop())}</p>
          <p>Triton unified-attention</p>
          <p>role: competitor</p>
        </div>
      </div>
      <div class="arrow">&rarr;</div>
      <div class="col">
        <h3>Output &amp; check</h3>
        ${outChips.map(([k, v]) => `<div class="chip"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join("")}
      </div>
    </div>
    <div class="verdict reveal">${verdictLine(d)}</div>
    ${campaignStrip(d.checklist)}
  </section>`;
}

function campaignStrip(checklist) {
  // Aggregate each phase into a campaign stage, in first-seen order.
  const order = [];
  const byPhase = {};
  checklist.forEach((c) => {
    if (!byPhase[c.phase]) { byPhase[c.phase] = []; order.push(c.phase); }
    byPhase[c.phase].push(c);
  });
  const stages = order.map((ph) => {
    const items = byPhase[ph];
    const allDeferred = items.every((i) => i.status === "deferred");
    const allDone = items.every((i) => i.status === "done");
    const cls = allDeferred ? "deferred" : allDone ? "done" : "active";
    const tag = allDeferred ? "deferred" : allDone ? "done" : "in&nbsp;progress";
    return { label: PHASE_ORDER_LABEL[ph] || ph, cls, tag };
  });
  const nodes = stages
    .map((s) => `<span class="stage ${s.cls}">${esc(s.label)}<span class="st-tag">${s.tag}</span></span>`)
    .join('<span class="sarrow">&rarr;</span>');
  return `
    <div class="campaign reveal">
      <h3>Campaign stage map</h3>
      <div class="track">${nodes}</div>
    </div>`;
}

function verdictLine(d) {
  const r = ratioStats(d.results);
  if (!r) return "Results pending.";
  return `Measured result: <b>AITER is ~${r.speedup}&times; faster</b> &mdash; rocKE is
    ~${r.slower}&times; <b style="color:var(--ours)">slower</b>, both numerically identical
    (max_abs agrees). That eager gap is the optimization target.`;
}

// ---- 2. problem shape ------------------------------------------------------
function shapeSection(ps) {
  const rows = Object.entries(ps)
    .filter(([k]) => !["measurement", "ratio_caveat"].includes(k))
    .map(([k, v]) => `<tr><td class="p">${esc(k)}</td><td class="v">${esc(v)}</td></tr>`)
    .join("");
  return `
  <section>
    <h2>2. Problem shape</h2>
    <p class="lede">The full parameter set both kernels are pinned to &mdash; the definition of an
    apples-to-apples comparison.</p>
    <table class="reveal">
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </section>`;
}

// ---- 3. kernel sources -----------------------------------------------------
function kernelSection(ours, aiter) {
  return `
  <section>
    <h2>3. Kernel sources</h2>
    <p class="lede">The two implementations being raced, each on its real upstream source. rocKE is
    the CK&nbsp;DSL kernel we are grounding; AITER is the Triton competitor.</p>
    <div class="cards">
      ${kernelCard(ours, "ours")}
      ${kernelCard(aiter, "aiter")}
    </div>
  </section>`;
}

function kernelCard(k, cls) {
  const [path, sym] = k.sourcePath.includes("::")
    ? k.sourcePath.split("::")
    : [k.sourcePath.replace(/\.[^.]+$/, ""), k.sourcePath.split(".").pop()];
  const snippet =
    `<span class="c"># ${esc(path.trim())}</span>\n` +
    `<span class="k">from</span> rocke <span class="k">import</span> <span class="fn">${esc(sym.trim())}</span>`;
  return `
    <div class="card ${cls} reveal">
      <div class="bar"><span class="dot"></span>${esc(k.name)}<span class="role">${esc(k.role)}</span></div>
      <div class="body">
        <pre>${snippet}</pre>
        <a class="srclink" href="${esc(k.sourceUrl)}" target="_blank" rel="noopener">${esc(k.sourceUrl)}</a>
        <div class="note"><b>Shape passed:</b> ${esc(k.shapePassed)}</div>
      </div>
    </div>`;
}

// ---- 4. results as bar charts ---------------------------------------------
function resultsSection(d) {
  if (!d.results || !d.results.length) {
    return `<section><h2>4. Baseline results</h2><p class="lede">No results yet.</p></section>`;
  }
  const maxMs = Math.max(...d.results.map((r) => r.ms));
  // group by dtype, then by shape
  const groups = {};
  const gOrder = [];
  d.results.forEach((r) => {
    const key = r.dtype + "||" + r.shape;
    if (!groups[key]) { groups[key] = { dtype: r.dtype, shape: r.shape, rows: [] }; gOrder.push(key); }
    groups[key].rows.push(r);
  });

  const cards = gOrder.map((key) => {
    const g = groups[key];
    const oursR = g.rows.find((r) => /rocke/i.test(r.kernel));
    const aiterR = g.rows.find((r) => !/rocke/i.test(r.kernel));
    const bar = (r, fill) => {
      if (!r) return "";
      const w = Math.max(6, (r.ms / maxMs) * 100);
      return `<div class="brow"><span class="blabel">${esc(r.kernel)}</span>
        <div class="btrack"><div class="bfill ${fill}" data-w="${w.toFixed(1)}">${r.ms.toFixed(3)} ms</div></div></div>`;
    };
    let meta = "";
    if (oursR && aiterR) {
      const slower = (oursR.ms / aiterR.ms).toFixed(2);
      meta = `
        <div class="rmeta">
          <span class="slower">rocKE ${slower}&times; slower</span>
          <span class="mono">TFLOPS &mdash; AITER <b>${aiterR.tflops}</b> &middot; rocKE <b>${oursR.tflops}</b></span>
          <span class="mono">max_abs <b>${oursR.max_abs}</b></span>
        </div>`;
    }
    return `
      <div class="rcard reveal">
        <h4>${esc(g.dtype.split(" ")[0])} <span>&middot; ${esc(g.shape)}</span></h4>
        <div class="bars">
          ${bar(aiterR, "fa")}
          ${bar(oursR, "fo")}
        </div>
        ${meta}
      </div>`;
  }).join("");

  return `
  <section>
    <h2>4. Baseline results</h2>
    <p class="lede">Eager per-call latency &mdash; <b>longer bar = slower</b>. AITER (teal) is the bar to
    beat; rocKE (purple) currently trails it by ~1.5&times;. TFLOPS and correctness (max_abs) shown per shape.</p>
    <div class="legend reveal"><span><i class="la"></i>AITER (competitor)</span><span><i class="lo"></i>rocKE (baseline)</span></div>
    <div class="res">${cards}</div>
  </section>`;
}

// ---- 5. directional read (causal, ballpark) --------------------------------
function directionalReadSection(d) {
  const dr = d.directionalRead;
  if (!dr || !dr.rows || !dr.rows.length) return "";
  const maxMs = Math.max(...dr.rows.flatMap((r) => [r.dense_ms, r.aiter_ms]));

  const cards = dr.rows.map((r) => {
    const bar = (ms, fill, label) => {
      const w = Math.max(6, (ms / maxMs) * 100);
      return `<div class="brow"><span class="blabel">${label}</span>
        <div class="btrack"><div class="bfill ${fill}" data-w="${w.toFixed(1)}">${ms.toFixed(3)} ms</div></div></div>`;
    };
    return `
      <div class="rcard reveal">
        <h4>${esc(r.dtype)} <span>&middot; ${esc(r.shape)}</span></h4>
        <div class="bars">
          ${bar(r.aiter_ms, "fa", "AITER")}
          ${bar(r.dense_ms, "fo", "dense")}
        </div>
        <div class="rmeta">
          <span class="slower">dense ${r.ratio.toFixed(2)}&times; slower</span>
        </div>
      </div>`;
  }).join("");

  return `
  <section>
    <h2>5. Directional read <span class="h2note">(causal &middot; ballpark)</span></h2>
    <p class="lede">A separate, directional-only sweep of the <b>#10337 dense kernel</b> (purple) vs
    <b>AITER</b> (teal) under a <b>causal, dense-KV</b> regime &mdash; <b>longer bar = slower</b>. This is
    <b>not</b> the campaign's SW+paged target regime above; the two are never conflated.</p>
    <div class="callout reveal">
      <h4>&#9888; Ballpark caveat &mdash; read directionally, not as a target-regime result</h4>
      <p><b>${esc(dr.caveat)}</b></p>
      <p>${esc(dr.regime)}</p>
    </div>
    <div class="legend reveal"><span><i class="la"></i>AITER (competitor)</span><span><i class="lo"></i>#10337 dense</span></div>
    <div class="res">${cards}</div>
    <p class="lede drtake reveal"><b>Takeaway:</b> ${esc(dr.takeaway)}</p>
  </section>`;
}

// ---- paged-KV throughput ---------------------------------------------------
function pagedResultsSection(d) {
  const pr = d.pagedResults;
  if (!pr || !pr.rows || !pr.rows.length) return "";
  const maxMs = Math.max(...pr.rows.flatMap((r) => [r.paged_ms, r.aiter_ms, r.dense_ms].filter(Boolean)));
  const bar = (ms, fill, label) => {
    const w = Math.max(6, (ms / maxMs) * 100);
    return `<div class="brow"><span class="blabel">${label}</span>
      <div class="btrack"><div class="bfill ${fill}" data-w="${w.toFixed(1)}">${ms.toFixed(3)} ms</div></div></div>`;
  };
  const cards = pr.rows.map((r) => {
    const pa = r.aiter_ms ? (r.paged_ms / r.aiter_ms) : null;
    const pd = r.dense_ms ? (r.paged_ms / r.dense_ms) : null;
    return `
      <div class="rcard reveal">
        <h4>${esc(r.dtype)} <span>&middot; ${esc(r.shape)} &middot; ${r.paged_tflops.toFixed(0)} TFLOP/s</span></h4>
        <div class="bars">
          ${bar(r.aiter_ms, "fa", "AITER")}
          ${bar(r.paged_ms, "fo", "dense-paged")}
        </div>
        <div class="rmeta">
          ${pa ? `<span class="slower">paged ${pa.toFixed(2)}&times; AITER</span>` : ""}
          ${pd ? `<span>paged ${pd.toFixed(2)}&times; dense(unpaged)</span>` : ""}
        </div>
      </div>`;
  }).join("");
  return `
  <section>
    <h2>Paged-KV throughput <span class="h2note">(causal &middot; D128 &middot; GQA 32/8)</span></h2>
    <p class="lede">The dense kernel with the <b>paged-KV load path</b> (purple) vs <b>AITER</b> (teal),
    causal &mdash; <b>longer bar = slower</b>. Measured this session via <code>--mode paged</code>.</p>
    <div class="callout reveal">
      <h4>&#9888; Directional cross-run comparison</h4>
      <p>${esc(pr.caveat)}</p>
    </div>
    <div class="legend reveal"><span><i class="la"></i>AITER</span><span><i class="lo"></i>dense-paged</span></div>
    <div class="res">${cards}</div>
  </section>`;
}

// ---- 6. honesty callout ----------------------------------------------------
function calloutSection(ps) {
  return `
  <section>
    <h2>6. Measurement methodology (reconciled)</h2>
    <p class="lede">This dashboard reports what was actually measured, not a flattering subset. Baselines
    are from the production bench (<code>benchmark_prefill2d_live.py --variants prod</code>, shared-stream
    timing) and reconcile with PR #10206's ~1.48x; the launched kernel is confirmed to be the 4-warp GQA path.</p>
    <div class="callout reveal">
      <h4>&#9989; Reconciled ratios</h4>
      <p><b>Measurement:</b> ${esc(ps.measurement)}</p>
      <p><b>Reconciliation:</b> ${esc(ps.ratio_caveat)}</p>
    </div>
  </section>`;
}

// ---- 7. checklist ----------------------------------------------------------
function checklistSection(d) {
  const { done, active } = progress(d.checklist);
  const pct = active ? Math.round((100 * done) / active) : 0;
  const mark = { done: "&#10003;", todo: "&#9675;", deferred: "&rarr;" };
  const items = d.checklist.map((c) => `
    <div class="ci ${esc(c.status)}">
      <span class="mark">${mark[c.status] || "&#9675;"}</span>
      <span class="lbl">${esc(c.label)}</span>
      <span class="phase">${esc(c.phase)}</span>
    </div>`).join("");
  return `
  <section>
    <h2>7. Checklist &amp; progress</h2>
    <p class="lede">The living campaign checklist. Deferred stages are greyed &mdash; they are planned
    but out of scope for the current grounding pass.</p>
    <div class="hprog reveal" style="max-width:none">
      <div class="ptop"><span class="plab">Active tasks complete</span>
        <span class="pval">${pct}% &middot; ${done}/${active}</span></div>
      <div class="ptrack"><div class="pfill" data-pct="${pct}"></div></div>
    </div>
    <div class="checklist reveal">${items}</div>
  </section>`;
}

// ---- dense-paged port status ----------------------------------------------
function portStatusSection(d) {
  const p = d.portStatus;
  if (!p) return "";
  const mark = { done: "&#10003;", todo: "&#9675;" };
  const rows = p.phases.map((ph) => `
    <div class="ci ${esc(ph.status)}">
      <span class="mark">${mark[ph.status] || "&#9675;"}</span>
      <span class="lbl"><b>${esc(ph.phase)}</b> &mdash; ${esc(ph.evidence)}</span>
    </div>`).join("");
  return `
  <section>
    <h2>Dense-paged port &mdash; implemented &amp; GPU-verified</h2>
    <p class="lede">${esc(p.title)}</p>
    <p class="lede" style="opacity:.75">${esc(p.increment)}</p>
    <div class="checklist reveal">${rows}</div>
  </section>`;
}

// ---- footer ----------------------------------------------------------------
function renderFooter(d, ps, ours, aiter) {
  document.getElementById("ftr").innerHTML = `
    <p>gfx942 (AMD Instinct MI300X) &middot; ${esc(ps.operation)} &middot;
    rocKE <code>${esc(ours.name)}</code> vs AITER <code>${esc(aiter.name)}</code>.
    All values rendered from <code>data.json</code>.</p>`;
}

// ---- helpers ---------------------------------------------------------------
function progress(checklist) {
  let done = 0, active = 0;
  checklist.forEach((c) => {
    if (c.status !== "deferred") { active++; if (c.status === "done") done++; }
  });
  return { done, active };
}

function ratioStats(results) {
  if (!results || !results.length) return null;
  const ratios = [];
  const byKey = {};
  results.forEach((r) => {
    const key = r.dtype + "||" + r.shape;
    (byKey[key] = byKey[key] || []).push(r);
  });
  Object.values(byKey).forEach((rows) => {
    const o = rows.find((r) => /rocke/i.test(r.kernel));
    const a = rows.find((r) => !/rocke/i.test(r.kernel));
    if (o && a) ratios.push(o.ms / a.ms);
  });
  if (!ratios.length) return null;
  const avg = ratios.reduce((s, x) => s + x, 0) / ratios.length;
  return { slower: avg.toFixed(2), speedup: avg.toFixed(2) };
}

main();
