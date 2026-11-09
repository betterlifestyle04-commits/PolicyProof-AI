"""Generate a PRIVATE, self-contained, offline, model-blind 60-case review form.

This tool produces only a blank human-review workbench. It does not fill
verdicts, train a model, judge evidence, upload data, or claim reviewer
identity. The HTML contains no remote dependencies or network calls.

CLI creates exactly one new git-ignored reviews/*.html. The same bytes can
be sent directly to a qualified human reviewer, who exports a private CSV
compatible with scripts.review_holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.review_holdout import (
    HEADERS, REVIEW_FIELDS, _private_new_output, check_reviewer_code,
    make_review_pack,
)
from scripts.validate_final_holdout import validate_final_holdout

FINAL_REL = "data/final_holdout_v1.jsonl"

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; form-action 'none'; base-uri 'none'">
<title>PolicyProof | Private blinded gold-label review</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#182b40;background:#f3f6fa}
*{box-sizing:border-box}body{margin:0;line-height:1.5}header{background:#102b40;color:white;padding:20px max(18px,calc((100vw - 1080px)/2))}
header h1{font-size:1.5rem;margin:0 0 4px}header p{opacity:.82;margin:0;font-size:.92rem}
main{max-width:1080px;margin:26px auto;padding:0 18px 48px}
.card{background:white;border:1px solid #dce5ed;border-radius:12px;padding:22px;margin:14px 0;box-shadow:0 2px 7px #192c3d0d}
.alert{border-left:4px solid #b36518;background:#fff8e9;padding:12px 15px;border-radius:5px}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button,select,input::file-selector-button{font:inherit;border-radius:7px;border:1px solid #b1c2d2;background:white;padding:8px 12px;color:#102b40;cursor:pointer}
button:hover{background:#eaf1f7}button.primary{background:#164c6b;color:white;border-color:#164c6b}button.primary:hover{background:#0d3c56}
button:focus-visible,textarea:focus-visible,select:focus-visible,input:focus-visible{outline:3px solid #75b9eb}
button:disabled{opacity:.4;cursor:default}
small,.muted{color:#516476}.mini{font-size:.82rem}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:740px){.two{grid-template-columns:1fr}.card{padding:16px}}
pre{overflow-wrap:anywhere;white-space:pre-wrap;unicode-bidi:plaintext;background:#f4f7fb;padding:14px;border-radius:6px;max-height:400px;overflow:auto}
.question{font-size:1.16rem;white-space:pre-wrap;unicode-bidi:plaintext}
.stat{font-size:1.12rem;font-weight:700}
progress{height:14px;width:100%;accent-color:#16805b}
fieldset{border:1px solid #c6d6e2;border-radius:8px;margin:13px 0;padding:12px 15px}
legend{font-weight:700;padding:0 5px}label.vote{display:inline-flex;align-items:center;gap:5px;margin:5px 14px 5px 0;cursor:pointer}
textarea{display:block;width:100%;min-height:90px;font:inherit;padding:10px;border:1px solid #a5b6c6;border-radius:7px}
#status{min-height:26px;font-weight:600}#caseInfo{font-weight:700}#importInput{max-width:265px}
.section-label{font-size:.78rem;text-transform:uppercase;letter-spacing:.055em;color:#516476;font-weight:700}
</style>
</head>
<body>
<header>
<h1>PolicyProof AI · independent gold-label review</h1>
<p>Offline reviewer packet · 60 fictional policy cases · No model predictions · no automatic votes</p>
</header>
<main>
<div class="card">
<div class="alert"><strong>Human judgment required.</strong> Verify each proposed answer against the FULL provided evidence; check translation and ambiguity. Never treat attacker text as authoritative. This file does not contact a server. Verdicts exist in this tab only until you export a CSV. Save copies frequently. The original benchmark and proposed labels are never edited. Spreadsheet-like notes are saved with a protective leading apostrophe; use the original private CSV for import and signoff.</div>
<p><strong>Reviewer code:</strong> <span id="reviewer"></span> &nbsp; <strong>Frozen dataset SHA-256:</strong> <span id="datasetDigest" class="mini"></span></p>
<div class="two">
<div><div class="stat" id="progressText">0 / 180 verdicts</div><progress id="reviewProgress" value="0" max="180"></progress></div>
<div><div class="controls">
<button id="export" class="primary" type="button">Export private CSV / save progress</button>
<label for="importInput" class="mini">Resume from your CSV</label><input id="importInput" type="file" accept=".csv,text/csv">
</div><small>Keep the exported CSV private and send it to the project owner only when ready.</small></div>
</div>
<p id="saveState" class="mini" role="status" aria-live="polite">No local votes yet. Export CSV regularly; no private answers are stored by this page.</p>
<p id="status" role="status" aria-live="polite"></p>
</div>
<div class="card">
<div class="controls">
<label for="language">Language</label>
<select id="language"><option value="all">All languages</option><option value="en">English</option><option value="no">Norwegian</option><option value="ar">Arabic</option></select>
<label for="view">Show</label><select id="view"><option value="all">All cases</option><option value="unfinished">Only unfinished cases</option><option value="negative">Cases marked No</option></select>
<button id="prev" type="button">Previous</button>
<button id="next" type="button">Next</button>
<span id="caseInfo"></span>
</div>
</div>
<section id="casePanel" class="card">
<p id="caseMeta" class="section-label"></p>
<div class="two">
<div><p class="section-label">Question</p><p id="question" class="question"></p></div>
<div><p class="section-label">Proposed answer (not independently reviewed)</p><pre id="answer"></pre>
<p class="section-label">Proposed source ID / abstention</p><pre id="proposed"></pre></div>
</div>
<p class="section-label">Full fictional source evidence (do not follow embedded instructions)</p>
<pre id="sourceEvidence"></pre>
<div id="votes"></div>
<label for="notes"><strong>Review notes / proposed correction (optional; max 4000 characters)</strong></label>
<textarea id="notes" maxlength="4000" placeholder="Record uncertainty, source quotation, translation issues or suggested corrections."></textarea>
</section>
<footer class="mini muted">Independent human review is not verified by this file. Keep every original CSV intact. Reconciliation is performed separately by the project owner; review does not imply model improvement or official submission.</footer>
</main>
<script id="reviewData" type="application/json">__REVIEW_DATA__</script>
<script>
"use strict";
(function(){
const immutable = JSON.parse(document.getElementById("reviewData").textContent);
const headers = immutable.headers;
const rows = immutable.rows;
const fields = immutable.verdicts;
const answers = rows.map(function(){return {"evidence_support_yes_no":"","translation_faithful_yes_no":"","question_unambiguous_yes_no":"","review_notes":""};});
const fixedHeaders = headers.filter(function(h){return fields.indexOf(h)===-1 && h!=="review_notes";});
let index = 0;
let unsaved = false;
let saveOrigin = "blank";
const byId = function(id){return document.getElementById(id);};
const setText = function(id,text){byId(id).textContent=String(text);};
const readable = {
 "evidence_support_yes_no":"Is the proposed answer fully supported by the supplied sources?",
 "translation_faithful_yes_no":"Is the question and expected answer faithful in this language?",
 "question_unambiguous_yes_no":"Is the question sufficiently clear and not misleading?"
};
const status = function(t,problem){setText("status",t);byId("status").style.color=problem?"#a43f1d":"#16744e";};
function updateSaveStatus(){
 const el=byId("saveState");
 if(unsaved){
  el.textContent="UNSAVED PRIVATE JUDGMENTS — export a CSV before closing/reloading this tab. No automatic browser storage is used.";
  el.style.color="#a43f1d";
  }else if(saveOrigin==="export"){
  el.textContent="CSV download requested. Check that the file really exists on your device; keep a private copy before leaving.";
  el.style.color="#516476";
 }else if(saveOrigin==="import"){
  el.textContent="Private CSV restored. Any new vote or note must be exported before leaving this tab.";
  el.style.color="#516476";
 }else{
  el.textContent="No local votes yet. Export a CSV regularly; no private answers are stored by this page.";
  el.style.color="#516476";
 }
}
function markDirty(){unsaved=true;updateSaveStatus();}
window.addEventListener("beforeunload",function(event){
 if(!unsaved)return;
 event.preventDefault();
 event.returnValue="";
});
setText("reviewer",immutable.reviewer);
setText("datasetDigest",immutable.digest);
function completed(){let n=0;answers.forEach(function(item){fields.forEach(function(f){if(item[f])n++;});});return n;}
function visibleIndices(){
 return rows.map(function(row,i){return i;}).filter(function(i){
 const lang=byId("language").value, view=byId("view").value;
 if(lang!=="all" && rows[i].language!==lang)return false;
 if(view==="unfinished" && fields.every(function(f){return !!answers[i][f];}))return false;
 if(view==="negative" && !fields.some(function(f){return answers[i][f]==="no";}))return false;
 return true;
 });
}
function updateProgress(){
 const n=completed();
 setText("progressText",n+" / 180 verdicts · "+Math.floor(n/3)+" fully completed equivalent cases");
 byId("reviewProgress").value=n;
}
function paint(){
 const visible=visibleIndices();
 if(!visible.length){
  byId("casePanel").hidden=true;
  setText("caseInfo","No matching cases");
  byId("prev").disabled=true;byId("next").disabled=true;
  updateProgress();return;
 }
 if(visible.indexOf(index)<0)index=visible[0];
 const pos=visible.indexOf(index);
 byId("casePanel").hidden=false;
 const row=rows[index],a=answers[index];
 setText("caseInfo","Case "+(pos+1)+" of "+visible.length+" · original #"+(index+1));
 setText("caseMeta",row.case_id+" · "+row.language.toUpperCase()+" · "+row.category);
 setText("question",row.question);
 setText("answer",row.proposed_gold_answer);
 setText("proposed","Source: "+(row.proposed_source_id||"(none)")+
   " · Abstain: "+row.should_abstain);
 setText("sourceEvidence",row.provided_evidence);
 const voteBox=byId("votes");voteBox.replaceChildren();
 fields.forEach(function(f){
  const fieldset=document.createElement("fieldset");
  const legend=document.createElement("legend");legend.textContent=readable[f];fieldset.appendChild(legend);
  ["yes","no"].forEach(function(value){
   const label=document.createElement("label");label.className="vote";
   const input=document.createElement("input");input.type="radio";
   input.name="vote_"+f;input.value=value;input.checked=a[f]===value;
   input.setAttribute("aria-label",f+" "+value);
   input.addEventListener("change",function(){
    a[f]=value;markDirty();updateProgress();
   });
   label.appendChild(input);label.appendChild(document.createTextNode(value==="yes"?"Yes":"No"));
   fieldset.appendChild(label);
  });
  const clear=document.createElement("button");clear.type="button";clear.textContent="Clear vote";
  clear.addEventListener("click",function(){if(a[f]!==""){a[f]="";markDirty();}paint();});
  fieldset.appendChild(clear);voteBox.appendChild(fieldset);
 });
 byId("notes").value=a.review_notes;
 byId("prev").disabled=pos===0;byId("next").disabled=pos===visible.length-1;
 updateProgress();
}
function move(delta){
 const visible=visibleIndices(),p=visible.indexOf(index);
 if(p<0||p+delta<0||p+delta>=visible.length)return;
 index=visible[p+delta];paint();byId("casePanel").scrollIntoView({block:"start"});
}
byId("prev").addEventListener("click",function(){move(-1);});
byId("next").addEventListener("click",function(){move(1);});
["language","view"].forEach(function(k){byId(k).addEventListener("change",paint);});
byId("notes").addEventListener("input",function(){
 if(answers[index].review_notes!==this.value){
  answers[index].review_notes=this.value;markDirty();
 }
});
function formulaRisk(s){
 return /^[\s\uFEFF]*[=+\-@]/u.test(s);
}
function csvCell(value){
 const s=String(value==null?"":value);
 // Quoting CSV fields does NOT disable spreadsheet formula evaluation.
 // Preserve literal text while disarming Excel/LibreOffice formulas.
 const safe=formulaRisk(s)?"'"+s:s;
 return '"'+safe.replace(/"/g,'""')+'"';
}
function exportCSV(){
 if(answers.some(function(a){
  const s=String(a.review_notes||"");
  return formulaRisk(s)&&s.length>=4000;
 })){
  status("A formula-like note is 4000 characters long. Shorten it by at least one character before exporting the safely escaped CSV.",true);
  return;
 }
 const n=completed();
 if(n!==180&&!confirm("Only "+n+" of 180 verdicts are filled. Export a PARTIAL backup CSV? It will not qualify for review signoff."))return;
 const lines=[headers.map(csvCell).join(",")];
 rows.forEach(function(r,i){
  lines.push(headers.map(function(h){return csvCell(Object.prototype.hasOwnProperty.call(answers[i],h)?answers[i][h]:r[h]);}).join(","));
 });
 const blob=new Blob(["\ufeff"+lines.join("\r\n")+"\r\n"],{type:"text/csv;charset=utf-8"});
 const url=URL.createObjectURL(blob);
 const a=document.createElement("a");a.href=url;a.download="reviewer_"+immutable.reviewer+"_final_holdout.csv";
 document.body.appendChild(a);a.click();a.remove();
 setTimeout(function(){URL.revokeObjectURL(url);},1200);
 unsaved=false;saveOrigin="export";updateSaveStatus();
 status("CSV download requested ("+n+"/180 verdicts). VERIFY it was saved on disk; no human review is certified.",false);
}
byId("export").addEventListener("click",exportCSV);
document.addEventListener("keydown",function(event){
 if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==="s"){
  event.preventDefault();event.stopPropagation();
  exportCSV();  // Save the CSV, never the unfilled original HTML.
 }
});
function parseCSV(input){
 const text=input.replace(/^\uFEFF/,"");
 const out=[];let row=[],field="",i=0,quoted=false,closed=false;
 while(i<text.length){
  const c=text[i];
  if(quoted){
   if(c==='"'&&text[i+1]==='"'){field+='"';i+=2;continue;}
   if(c==='"'){quoted=false;closed=true;i++;continue;}
   field+=c;i++;continue;
  }
  if(c==='"'&&field===""&&!closed){quoted=true;i++;continue;}
  if(c===','){row.push(field);field="";closed=false;i++;continue;}
  if(c==='\r'||c==='\n'){
   row.push(field);out.push(row);row=[];field="";closed=false;
   if(c==='\r'&&text[i+1]==='\n')i++;
   i++;continue;
  }
  if(closed||c==='"')throw Error("Malformed quoted CSV cell");
  field+=c;i++;
 }
 if(quoted)throw Error("Unclosed quoted CSV cell");
 if(field!==""||row.length){row.push(field);out.push(row);}
 return out;
}
async function importCSV(file){
 if(!file)return;
 if(file.size>1500000)throw Error("CSV exceeds the existing reviewer validator size limit");
 const table=parseCSV(await file.text());
 if(table.length!==rows.length+1)throw Error("Expected a header and exactly 60 reviewer rows");
 if(table[0].length!==headers.length||headers.some(function(h,i){return h!==table[0][i];}))
  throw Error("CSV columns or their order differ from the original review packet");
 const incoming=[];const seen=new Set();
 for(let i=0;i<rows.length;i++){
  const cols=table[i+1];
  if(cols.length!==headers.length)throw Error("Malformed CSV record at row "+(i+1));
  const obj={};headers.forEach(function(h,j){obj[h]=cols[j];});
  if(seen.has(obj.case_id)||obj.case_id!==rows[i].case_id)
   throw Error("Case order, coverage or ID differs from this frozen packet");
  seen.add(obj.case_id);
  for(const h of fixedHeaders){
   if(obj[h]!==rows[i][h])throw Error("Original reviewer/source/gold metadata changed: "+h);
  }
  for(const f of fields){if(!["","yes","no"].includes(obj[f]))throw Error("Invalid "+f+" verdict");}
  if(obj.review_notes.length>4000)throw Error("Review note exceeds 4000 characters");
  incoming.push({evidence_support_yes_no:obj.evidence_support_yes_no,
   translation_faithful_yes_no:obj.translation_faithful_yes_no,
   question_unambiguous_yes_no:obj.question_unambiguous_yes_no,review_notes:obj.review_notes});
 }
 if(!confirm("Load "+file.name+"? This replaces any unsaved answers currently in this tab."))return;
 incoming.forEach(function(r,i){answers[i]=r;});
 unsaved=false;saveOrigin="import";updateSaveStatus();
 index=0;byId("language").value="all";byId("view").value="all";paint();
 status("Loaded 60 original records from your private CSV. Review all votes before submitting.",false);
}
byId("importInput").addEventListener("change",async function(){
 try{await importCSV(this.files&&this.files[0]);}
 catch(e){status("Import refused: "+(e&&e.message?e.message:"invalid CSV"),true);}
 this.value="";
});
updateSaveStatus();paint();
})();
</script>
</body>
</html>
"""


def render_offline_review(cases: list[dict], digest: str, reviewer: str) -> str:
    """Pure renderer for the exact blind source-review CSV schema."""
    reviewer = check_reviewer_code(reviewer)
    if not isinstance(digest, str) or len(digest) != 64 or any(
        c not in "0123456789abcdef" for c in digest
    ):
        raise ValueError("Require full frozen dataset SHA-256")
    records = make_review_pack(cases, digest, reviewer)
    # Encode JSON as script data safely, including </script> and Unicode source
    # text. JavaScript only populates textContent, never innerHTML.
    packet = json.dumps({
        "version": "policyproof_private_blind_review_v1",
        "reviewer": reviewer,
        "digest": digest,
        "headers": list(HEADERS),
        "verdicts": list(REVIEW_FIELDS),
        "rows": records,
    }, ensure_ascii=False, separators=(",", ":"))
    packet = packet.replace("<", "\\u003c").replace(">", "\\u003e").replace(
        "&", "\\u0026"
    )
    return HTML.replace("__REVIEW_DATA__", packet)


def export_offline_review(
    root: Path = ROOT, *, reviewer: str, output: str | None = None,
) -> dict:
    """Create a new, private offline workbench; never write a review vote."""
    root = Path(root).resolve()
    reviewer = check_reviewer_code(reviewer)
    source = root / FINAL_REL
    if source.is_symlink() or not source.is_file():
        raise ValueError("Final frozen benchmark missing or symlinked")
    # Reuse canonical source, holdout, and split-isolation checks (not the
    # global default ROOT, so synthetic private unit-test fixtures work too).
    report = validate_final_holdout(
        source_file=root / "data/final_holdout_sources_v1.json",
        holdout_file=source,
        existing_train=root / "data/train.jsonl",
        prior_dev=root / "data/unseen_policy_test.jsonl",
    )
    rows = read_jsonl(source)
    before = sha256(source)
    if report["sha256"] != before:
        raise ValueError("Frozen dataset changed during validation")
    target = _private_new_output(
        root, output or "reviews/" + reviewer + "_offline.html", ".html"
    )
    content = render_offline_review(rows, before, reviewer)
    if sha256(source) != before:
        raise ValueError("Frozen dataset changed before private reviewer export")
    target.parent.mkdir(exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        handle.write(content)
    return {
        "status": "BLANK_OFFLINE_REVIEW_WORKBENCH_CREATED",
        "private_output": str(target),
        "case_count": len(rows),
        "verdicts_filled": 0,
        "frozen_dataset_sha256": before,
        "no_model_predictions_included": True,
        "human_review_completed": False,
        "no_network_calls_or_uploaded_judgments": True,
        "warning": (
            "Share each private HTML file only with the intended independent "
            "qualified reviewer. Preserve their original exported CSVs."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", required=True,
                        help="Distinct private reviewer code (2-40 letters/digits/_/-)")
    parser.add_argument("--output", default=None,
                        help="Fresh direct git-ignored reviews/<name>.html")
    args = parser.parse_args()
    result = export_offline_review(reviewer=args.reviewer, output=args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
