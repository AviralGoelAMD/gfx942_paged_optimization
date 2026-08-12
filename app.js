const STAGES = [
  {k:"shape",label:"Fix shape"},{k:"competitor",label:"AITER competitor"},
  {k:"baseline",label:"Baseline"},{k:"port",label:"Port dense-paged",deferred:true},
  {k:"3way",label:"3-way",deferred:true},{k:"isa",label:"ISA",deferred:true},
  {k:"pr",label:"Routing PR",deferred:true}
];
async function main(){
  const d = await (await fetch("data.json?"+Date.now())).json();
  document.getElementById("title").textContent = d.title;
  document.getElementById("updated").textContent = "Updated "+d.updated;
  // stage map
  const sm = document.querySelector("#stage-map .stages");
  STAGES.forEach((s,i)=>{
    const el=document.createElement("span");el.className="stage"+(s.deferred?" deferred":"");
    el.textContent=s.label;sm.appendChild(el);
    if(i<STAGES.length-1){const a=document.createElement("span");a.className="arrow";a.textContent="\u2192";sm.appendChild(a);}
  });
  // shape
  const st=document.querySelector("#shape table");
  for(const [k,v] of Object.entries(d.problemShape)){
    st.insertAdjacentHTML("beforeend",`<tr><th>${k}</th><td>${v}</td></tr>`);
  }
  // kernels
  const cards=document.querySelector("#kernels .cards");
  d.kernels.forEach(kn=>{
    const caveat = /parity|caveat|PLACEHOLDER/i.test(kn.shapePassed)?`<div class="note caveat">${kn.shapePassed}</div>`:`<div class="note"><b>Shape passed:</b> ${kn.shapePassed}</div>`;
    cards.insertAdjacentHTML("beforeend",
      `<div class="card"><div class="bar">${kn.name} <small>(${kn.role})</small></div>
       <div class="body"><div>${kn.sourcePath}</div><a href="${kn.sourceUrl}">${kn.sourceUrl}</a>${caveat}</div></div>`);
  });
  // results
  const rt=document.querySelector("#results table"), empty=document.querySelector("#results .empty");
  if(d.results.length){
    empty.style.display="none";
    rt.insertAdjacentHTML("beforeend","<tr><th>shape</th><th>dtype</th><th>kernel</th><th>ms</th><th>TFLOPS</th><th>max_abs</th><th>notes</th></tr>");
    d.results.forEach(r=>rt.insertAdjacentHTML("beforeend",
      `<tr><td>${r.shape}</td><td>${r.dtype}</td><td>${r.kernel}</td><td>${r.ms}</td><td>${r.tflops}</td><td>${r.max_abs}</td><td>${r.notes||""}</td></tr>`));
  }
  // checklist + progress
  const ul=document.querySelector("#checklist ul");let done=0,active=0;
  d.checklist.forEach(c=>{
    ul.insertAdjacentHTML("beforeend",`<li class="st-${c.status}">[${c.id}] ${c.label}</li>`);
    if(c.status!=="deferred"){active++; if(c.status==="done") done++;}
  });
  const pct = active?Math.round(100*done/active):0;
  document.getElementById("progress-bar").style.width=pct+"%";
  document.getElementById("progress-label").textContent=pct+"% ("+done+"/"+active+")";
}
main();
