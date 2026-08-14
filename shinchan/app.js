// shinchan — fp16 causal dense-paged prefill vs AITER (gfx942) dashboard.
// 100% data-driven: every value below is read from data.json at load time.
// Perf / counter / ISA sections render a visible "measurement pending"
// placeholder until the coordinator populates results.rows, counters.rows,
// isa.ours and isa.aiter.

const esc = (s) => String(s).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// human labels for the problem-shape table keys
const SHAPE_LABEL = {
  dtype: "dtype",
  mask: "mask",
  head_dim: "head_dim",
  num_query_heads: "num_query_heads (Hq)",
  num_kv_heads: "num_kv_heads (Hkv)",
  gqa_ratio: "gqa_ratio",
  kv_layout: "kv_layout",
  block_size: "block_size",
  num_seqs: "num_seqs",
  seqlens: "seqlens",
  scale: "scale",
  softcap_alibi_sinks: "softcap / alibi / sinks",
};

const STATUS_MARK = { done: "&#10003;", in_progress: "&#8226;", pending: "&#9675;" };
const STATUS_TAG = { done: "done", in_progress: "in&nbsp;progress", pending: "pending" };

async function main() {
  const d = await (await fetch("data.json?" + Date.now())).json();
  const ps = d.problemShape;
  const ours = d.kernels.find((k) => k.role === "ours");
  const aiter = d.kernels.find((k) => k.role === "competitor");

  document.title = d.title;
  renderHeader(d, ps);
  document.getElementById("main").innerHTML =
    shapeSection(ps) +
    kernelSection(d, ours, aiter) +
    resultsSection(d) +
    countersSection(d) +
    isaSection(d) +
    checklistSection(d);
  renderFooter(d, ours, aiter);

  requestAnimationFrame(() => {
    document.querySelectorAll(".pfill").forEach((p) => (p.style.width = p.dataset.pct + "%"));
  });

  const io = new IntersectionObserver(
    (es) => es.forEach((e) => { if (e.isIntersecting) e.target.classList.add("in"); }),
    { threshold: 0.12 }
  );
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
}

// ---- header ----------------------------------------------------------------
function renderHeader(d, ps) {
  const { done, total } = progress(d.checklist);
  const pct = total ? Math.round((100 * done) / total) : 0;
  document.getElementById("hdr").innerHTML = `
    <h1>${esc(d.title)}</h1>
    <div class="sub">Living dashboard for the <b>fp16 causal dense-paged prefill</b> optimization
    campaign &mdash; head&nbsp;dim&nbsp;${esc(ps.head_dim)}, ${esc(ps.mask)}, GQA
    ${esc(ps.num_query_heads)}/${esc(ps.num_kv_heads)}. We ground our CK&nbsp;DSL kernel
    (<b>rockKE dense-paged</b>) against <b>AITER unified_attention</b> on an identical problem shape,
    then profile counters and ISA to find the next lever.</div>
    <div class="tags">
      <span class="tag">dtype = ${esc(ps.dtype)}</span>
      <span class="tag">${esc(ps.mask)}</span>
      <span class="tag ours">rockKE = purple</span>
      <span class="tag aiter">AITER = teal</span>
    </div>
    <div class="hprog">
      <div class="ptop"><span class="plab">Campaign progress</span>
        <span class="pval">${pct}% &middot; ${done}/${total} done</span></div>
      <div class="ptrack"><div class="pfill" data-pct="${pct}"></div></div>
    </div>
    <div class="updated">Updated ${esc(d.updated)}</div>`;
}

// ---- 1. problem shape ------------------------------------------------------
function shapeSection(ps) {
  const rows = Object.entries(ps)
    .map(([k, v]) => `<tr><td class="p">${esc(SHAPE_LABEL[k] || k)}</td><td class="v">${esc(v)}</td></tr>`)
    .join("");
  return `
  <section>
    <h2>1. Problem shape</h2>
    <p class="lede">The full parameter set both kernels are pinned to &mdash; the definition of an
    apples-to-apples comparison.</p>
    <table class="reveal">
      <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </section>`;
}

// ---- 2. kernel sources -----------------------------------------------------
function kernelSection(d, ours, aiter) {
  return `
  <section>
    <h2>2. Kernel sources</h2>
    <p class="lede">The two implementations being raced, each on its real upstream source. rockKE is
    the CK&nbsp;DSL kernel we are grounding; AITER is the Triton competitor.</p>
    <div class="cards">
      ${kernelCard(ours, "ours")}
      ${kernelCard(aiter, "aiter")}
    </div>
    <div class="callout reveal" style="background:var(--accbg);border-color:#17456f;border-left-color:var(--acc)">
      <h4 style="color:var(--acc)">Apples-to-apples</h4>
      <p style="color:#cfe0f2">${esc(d.applesToApples)}</p>
    </div>
  </section>`;
}

function kernelCard(k, cls) {
  return `
    <div class="card ${cls} reveal">
      <div class="bar"><span class="dot"></span>${esc(k.name)}<span class="role">${esc(k.role)}</span></div>
      <div class="body">
        ${k.sourcePath ? `<pre><span class="c"># ${esc(k.sourcePath)}</span></pre>` : ""}
        <a class="srclink" href="${esc(k.sourceUrl)}" target="_blank" rel="noopener">${esc(k.sourceUrl)}</a>
        <div class="note"><b>Args passed:</b> ${esc(k.argsPassed)}</div>
      </div>
    </div>`;
}

// ---- pending placeholder ---------------------------------------------------
function pendingBox(note) {
  return `
    <div class="callout pending reveal">
      <h4>&#9203; Measurement pending</h4>
      <p>${esc(note || "This section will populate once the profiling data is available.")}</p>
    </div>`;
}

// ---- 3. performance --------------------------------------------------------
function resultsSection(d) {
  const r = d.results || {};
  let body;
  if (!r.rows || !r.rows.length) {
    body = pendingBox(r.note);
  } else {
    const maxMs = Math.max(...r.rows.map((row) => Number(row.ours_ms) || 0).concat(
      r.rows.map((row) => Number(row.aiter_ms) || 0)));
    const bar = (ms, fill, label) => {
      const w = maxMs ? Math.max(6, (ms / maxMs) * 100) : 6;
      return `<div class="brow"><span class="blabel">${label}</span>
        <div class="btrack"><div class="bfill ${fill}" data-w="${w.toFixed(1)}" style="width:${w.toFixed(1)}%">${Number(ms).toFixed(3)} ms</div></div></div>`;
    };
    const cards = r.rows.map((row) => {
      const ratio = (Number(row.ours_ms) && Number(row.aiter_ms))
        ? (Number(row.ours_ms) / Number(row.aiter_ms)).toFixed(2) : null;
      return `
        <div class="rcard reveal">
          <h4>${esc(row.dtype || "")} <span>&middot; ${esc(row.shape || "")}</span></h4>
          <div class="bars">
            ${bar(row.aiter_ms, "fa", "AITER")}
            ${bar(row.ours_ms, "fo", "rockKE")}
          </div>
          ${ratio ? `<div class="rmeta"><span class="slower">rockKE ${ratio}&times; vs AITER</span>
            ${row.max_abs != null ? `<span class="mono">max_abs <b>${esc(row.max_abs)}</b></span>` : ""}</div>` : ""}
        </div>`;
    }).join("");
    body = `<div class="legend reveal"><span><i class="la"></i>AITER (competitor)</span><span><i class="lo"></i>rockKE (ours)</span></div>
      <div class="res">${cards}</div>`;
  }
  return `
  <section>
    <h2>3. Performance ${statusNote(r)}</h2>
    <p class="lede">Per-call latency &mdash; <b>longer bar = slower</b>. AITER (teal) is the bar to beat;
    rockKE is purple. Populated after the rocprof timing run.</p>
    ${body}
  </section>`;
}

// ---- 4. hardware-counter comparison ---------------------------------------
function countersSection(d) {
  const c = d.counters || {};
  let body;
  if (!c.rows || !c.rows.length) {
    body = pendingBox(c.note);
  } else {
    const head = `<tr><th>Counter</th><th>rockKE</th><th>AITER</th><th>note</th></tr>`;
    const rows = c.rows.map((row) => {
      const def = row.def ? ` title="${esc(row.def)}" style="cursor:help;border-bottom:1px dotted currentColor"` : "";
      return `
      <tr><td class="p"><span${def}>${esc(row.counter || "")}</span></td>
      <td class="v">${esc(row.ours != null ? row.ours : "")}</td>
      <td class="v">${esc(row.aiter != null ? row.aiter : "")}</td>
      <td class="v">${esc(row.note || "")}</td></tr>`;
    }).join("");
    body = `<table class="reveal"><thead>${head}</thead><tbody>${rows}</tbody></table>`;
  }
  return `
  <section>
    <h2>4. Hardware-counter comparison ${statusNote(c)}</h2>
    <p class="lede">rocprof hardware counters for both kernels on the identical shape &mdash; the
    evidence used to locate the bottleneck.</p>
    ${body}
  </section>`;
}

// ---- 5. ISA comparison -----------------------------------------------------
function isaSection(d) {
  const i = d.isa || {};
  let body;
  if (!i.ours && !i.aiter) {
    body = pendingBox(i.note);
  } else {
    const panel = (title, cls, text) => `
      <div class="card ${cls} reveal">
        <div class="bar"><span class="dot"></span>${title}</div>
        <div class="body">${text ? `<pre>${esc(text)}</pre>` : pendingBox("ISA dump not yet captured.")}</div>
      </div>`;
    body = `<div class="cards">
      ${panel("rockKE ISA", "ours", i.ours)}
      ${panel("AITER ISA", "aiter", i.aiter)}
    </div>`;
  }
  return `
  <section>
    <h2>5. ISA comparison ${statusNote(i)}</h2>
    <p class="lede">Disassembled hot loops for both kernels &mdash; instruction mix, VMEM/LDS traffic and
    occupancy limiters read straight off the ISA.</p>
    ${body}
  </section>`;
}

// ---- 6. checklist ----------------------------------------------------------
function checklistSection(d) {
  const { done, total } = progress(d.checklist);
  const pct = total ? Math.round((100 * done) / total) : 0;
  const items = d.checklist.map((c) => `
    <div class="ci ${esc(c.status)}">
      <span class="mark">${STATUS_MARK[c.status] || "&#9675;"}</span>
      <span class="lbl">${esc(c.label)}</span>
      <span class="phase">${STATUS_TAG[c.status] || esc(c.status)}</span>
    </div>`).join("");
  return `
  <section>
    <h2>6. Checklist &amp; progress</h2>
    <p class="lede">The living campaign checklist, from framing the problem through implementing and
    verifying the optimization.</p>
    <div class="hprog reveal" style="max-width:none">
      <div class="ptop"><span class="plab">Steps complete</span>
        <span class="pval">${pct}% &middot; ${done}/${total}</span></div>
      <div class="ptrack"><div class="pfill" data-pct="${pct}"></div></div>
    </div>
    <div class="checklist reveal">${items}</div>
  </section>`;
}

// ---- footer ----------------------------------------------------------------
function renderFooter(d, ours, aiter) {
  document.getElementById("ftr").innerHTML = `
    <p>gfx942 (AMD Instinct MI300X) &middot; fp16 causal dense-paged prefill &middot;
    rockKE <code>${esc(ours.name)}</code> vs AITER <code>${esc(aiter.name)}</code>.
    All values rendered from <code>data.json</code>.</p>`;
}

// ---- helpers ---------------------------------------------------------------
function statusNote(section) {
  if (!section || !section.status || section.status === "done") return "";
  const label = section.status === "in_progress" ? "in progress" : esc(section.status);
  return `<span class="h2note">(${label})</span>`;
}

function progress(checklist) {
  let done = 0;
  const total = checklist.length;
  checklist.forEach((c) => { if (c.status === "done") done++; });
  return { done, total };
}

main();
