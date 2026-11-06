#!/usr/bin/env node
'use strict';

// Local, read-only pair browser. Node >=18; no packages, providers, or credentials.
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const crypto = require('node:crypto');
const ROOT = path.resolve(__dirname, '..');
const CATALOG = 'research_runs/pair_review_v1/catalog.private.json';
const sha = text => crypto.createHash('sha256').update(text, 'utf8').digest('hex');

function localPath(root, name) {
  const absolute = path.resolve(root, name);
  if (!absolute.startsWith(path.resolve(root) + path.sep)) throw Error('Catalog path must stay inside the project');
  if (fs.existsSync(absolute) && !fs.realpathSync(absolute).startsWith(fs.realpathSync(root) + path.sep)) throw Error('External symlinks are not supported');
  return absolute;
}
function json(root, name) { return JSON.parse(fs.readFileSync(localPath(root, name), 'utf8')); }
function jsonl(root, name) {
  return fs.readFileSync(localPath(root, name), 'utf8').split(/\r?\n/).filter(s => s.trim()).map((s, i) => {
    try { return JSON.parse(s); } catch { throw Error(`${name}: invalid JSON at line ${i + 1}`); }
  });
}
function value(row, names) {
  for (const name of names) {
    let v = row;
    for (const key of name.split('.')) v = v?.[key];
    if (v !== undefined && v !== null && v !== '') return v;
  }
  return '';
}

function defaultCatalog() {
  return {
    version: 1,
    description: 'Existing pairs only. Training has matched targets; Gemini held-out coverage is incomplete. Test bodies load only when Test is selected.',
    paperRegistries: ['docs/author_inventory_v3/inventory.json', 'data/author_style_corpus_v3/source_registry.private.json', 'data/author_style_corpus_v3_existing/source_registry.private.json'],
    paperAliases: 'research_runs/data_scaling_v1/gpt6_track_0500_v1/paper_identity_audit.json',
    targetIssues: 'research_runs/data_scaling_v1/gpt6_track_0500_v1/response_quality_v1/adjudications.private.jsonl',
    datasets: [
      { id: 'gpt6-training-500', label: 'GPT-6 matched 500 candidates', adapter: 'gpt6_candidates', split: 'training', teacher: 'GPT-6', model: 'GPT-6 (reported/requested)',
        path: 'research_runs/data_scaling_v1/gpt6_track_0500_v1/response_quality_v1/candidates_0500.private.jsonl',
        reviewed: 'research_runs/data_scaling_v1/gpt6_track_0500_v1/response_quality_v1/reviewed_responses.private.jsonl',
        lineage: 'research_runs/data_scaling_v1/gpt6_track_0500_v1/target_lineage.private.jsonl', status: 'Candidate; not a training release' },
      { id: 'gemini-training-500', label: 'Gemini matched 500 staging', adapter: 'gemini_checkpoint', split: 'training', teacher: 'Gemini', model: 'gemini-3-flash-preview',
        targets: 'research_runs/data_scaling_v1/gemini_track_0500_v1/targets.private.jsonl',
        drafts: 'research_runs/data_scaling_v1/gemini_track_0500_v1/drafts.private.jsonl',
        qualified: 'research_runs/data_scaling_v1/gemini_track_0500_v1/resume_20261105_v2/qualified_completed_reviews_v2.private.jsonl',
        lineage: 'research_runs/data_scaling_v1/gpt6_track_0500_v1/target_lineage.private.jsonl', status: 'Staging; 46 reviews pending' },
      { id: 'gpt6-development-salvage', label: 'GPT-6 curated development', adapter: 'pairs', split: 'development', teacher: 'GPT-6', model: 'GPT-6 (reported)',
        path: 'data/author_pairs_salvage_v1/development.private.jsonl', status: 'Earlier automated pair acceptance' },
      { id: 'pilot-development', label: 'Mixed-500 pilot held-out inputs', adapter: 'pairs', split: 'development', teacher: 'GPT-6', model: 'GPT-6 (reported)',
        path: 'data/corpus_500_pilot_v1/upload/corpus_500_pairs.jsonl', where: { split: 'development' },
        modelByCondition: { gemini_input: 'gemini-3-flash-preview', existing_input: 'GPT-6 (reported)' },
        teacherByCondition: { gemini_input: 'Gemini', existing_input: 'GPT-6' }, status: 'Previously used development input' },
      { id: 'legacy-fama-validation', label: 'Legacy Fama validation (21)', adapter: 'pairs', split: 'development', teacher: 'GPT-6', model: 'GPT-6 (user-confirmed historical workflow)',
        path: 'data/fama_corpus_v1/validation.jsonl', status: 'Historical validation; not a fresh holdout' },
      { id: 'legacy-fama-test', label: 'Legacy Fama reserved test (45)', adapter: 'pairs', split: 'test', teacher: 'GPT-6', model: 'GPT-6 (user-confirmed historical workflow)',
        path: 'data/fama_corpus_v1/test.jsonl', declaredCount: 45, status: 'Reserved historical test; no Gemini counterpart collected' },
    ],
  };
}

function readCatalog(root, name = CATALOG) {
  const catalog = fs.existsSync(localPath(root, name)) ? json(root, name) : defaultCatalog();
  if (catalog.version !== 1 || !Array.isArray(catalog.datasets)) throw Error('Unsupported catalog version');
  if (new Set(catalog.datasets.map(d => d.id)).size !== catalog.datasets.length) throw Error('Duplicate dataset IDs');
  for (const d of catalog.datasets) {
    if (!['training', 'development', 'test'].includes(d.split)) throw Error('Declare each dataset split explicitly');
    if (!['pairs', 'gpt6_candidates', 'gemini_checkpoint'].includes(d.adapter)) throw Error('Unsupported dataset adapter');
  }
  return catalog;
}

function metadata(root, catalog) {
  const byId = new Map(), byPdf = new Map(), issues = new Map();
  for (const file of catalog.paperRegistries || []) {
    if (!fs.existsSync(localPath(root, file))) continue;
    for (const paper of json(root, file).papers || []) {
      byId.set(paper.paper_id, paper);
      if (paper.pdf_sha256) byPdf.set(paper.pdf_sha256, paper);
    }
  }
  if (catalog.paperAliases && fs.existsSync(localPath(root, catalog.paperAliases))) {
    for (const group of json(root, catalog.paperAliases).alias_groups || []) {
      const paper = byPdf.get(group.pdf_sha256) || group.paper_ids.map(i => byId.get(i)).find(Boolean) || {};
      for (const id of group.paper_ids) byId.set(id, { ...paper, pdf_sha256: group.pdf_sha256 });
    }
  }
  if (catalog.targetIssues && fs.existsSync(localPath(root, catalog.targetIssues))) {
    for (const r of jsonl(root, catalog.targetIssues)) if (r.decision === 'target_review_pending') issues.set(r.target_sha256, `${r.id}: ${r.reason}`);
  }
  return { byId, byPdf, issues };
}

function normalizePair(row, dataset, meta, extra = {}) {
  const source = value(row, dataset.sourceFields || ['source', 'plain_english']);
  const target = value(row, dataset.targetFields || ['target', 'original_paragraph', 'reference', 'completion']);
  if (typeof source !== 'string' || !source.trim() || typeof target !== 'string' || !target.trim()) throw Error('Pair has missing source/target text');
  const sourceSha = sha(source), targetSha = sha(target);
  if (row.source_sha256 && row.source_sha256 !== sourceSha) throw Error('Source hash mismatch');
  if (row.target_sha256 && row.target_sha256 !== targetSha) throw Error('Target hash mismatch');
  const paperId = row.paper_id || row.paper_key || 'Unspecified paper';
  const pdf = extra.pdf || value(row, ['provenance.canonical_pdf_sha256', 'provenance.pdf_sha256', 'pdf_sha256']);
  const paper = meta.byPdf.get(pdf) || meta.byId.get(paperId) || {};
  const authors = Array.isArray(row.authors) ? row.authors : row.authors ? [String(row.authors)] : paper.authors || [];
  let group = row.style_group || row.author_group_id;
  if (!group) group = /Fama/i.test(authors.join(' ')) ? 'Fama_coauthors' : authors.join('; ') || 'Unspecified author';
  let teacher = extra.teacher || dataset.teacherByCondition?.[row.input_condition] || dataset.teacher || value(row, ['provenance.teacher_track']) || 'Unknown';
  if (row.provenance?.new_neutral_input_author === 'Codex') teacher = 'Codex';
  const id = extra.shortId || row.id || row.example_id || row.paragraph_id || targetSha.slice(0, 16);
  const canonicalPaper = pdf || paper.pdf_sha256 || paperId;
  const key = sha([dataset.split, canonicalPaper, teacher, sourceSha, targetSha].join('\n'));
  const targetIssue = meta.issues.get(targetSha) || '';
  const repairs = extra.repairs || [];
  return {
    key, feedbackId: 'pair-' + key.slice(0, 16), id, originalId: row.id || row.example_id || row.paragraph_id || id,
    datasetId: dataset.id, datasets: [dataset.label || dataset.id], split: dataset.split,
    teacher, model: extra.model || dataset.modelByCondition?.[row.input_condition] || dataset.model || value(row, ['provenance.generator', 'model']) || 'Not recorded',
    paperId, paperKey: canonicalPaper, title: row.title || paper.title || paperId,
    year: row.year || paper.version_year || paper.publication_year || '', authors, group,
    source, target, sourceSha, targetSha, rawSource: extra.rawSource || source, repairs,
    status: extra.status || dataset.status || value(row, ['quality.text_pair_status']) || 'Unspecified',
    pending: Boolean(extra.pending), repaired: repairs.length > 0, targetIssue,
    sourceFile: dataset.path || dataset.drafts, sourceLocation: row.provenance?.page_location || row.page_location || null,
  };
}

function loadDataset(root, d, meta) {
  if (d.adapter === 'gemini_checkpoint') {
    const drafts = new Map(jsonl(root, d.drafts).map(r => [r.id, r]));
    const decisions = new Map(jsonl(root, d.qualified).map(r => [r.id, r]));
    const lineage = new Map(jsonl(root, d.lineage).map(r => [r.mixed_index, r]));
    return jsonl(root, d.targets).map(t => {
      const raw = drafts.get(t.id), decision = decisions.get(t.id);
      if (!raw || sha(t.target) !== t.target_sha256) throw Error('Gemini target/draft binding mismatch');
      const rawSha = sha(raw.plain_english);
      if (raw.source_sha256 !== rawSha || (decision && (decision.draft_sha256 !== rawSha || decision.target_sha256 !== t.target_sha256))) throw Error('Gemini decision binding mismatch');
      const source = decision ? decision.final_source : raw.plain_english;
      if (decision && decision.source_sha256 !== sha(source)) throw Error('Gemini final source hash mismatch');
      const accepted = !raw.review_required || Boolean(decision);
      return normalizePair({ ...t, source, source_sha256: sha(source), authors: meta.byId.get(t.paper_id)?.authors || [] }, d, meta,
        { pdf: lineage.get(t.index)?.canonical_pdf_sha256, shortId: t.id, rawSource: raw.plain_english,
          repairs: decision?.applied_patches || [], pending: !accepted,
          status: accepted ? 'Input automatically qualified; staging' : 'Draft — separate review pending' });
    });
  }
  const reviewed = d.reviewed ? new Map(jsonl(root, d.reviewed).map(r => [r.id, r])) : new Map();
  const lineage = d.lineage ? new Map(jsonl(root, d.lineage).map(r => [r.mixed_id, r])) : new Map();
  return jsonl(root, d.path).filter(r => !d.where || Object.entries(d.where).every(([k, v]) => r[k] === v)).map(row => {
    const entry = lineage.get(row.provenance?.matched_mixed_id);
    const review = reviewed.get(entry?.id);
    if (review && (review.target_sha256 !== row.target_sha256 || review.source_sha256 !== row.source_sha256)) throw Error('GPT-6 reviewed candidate binding mismatch');
    return normalizePair(row, d, meta, {
      shortId: entry?.id, pdf: entry?.canonical_pdf_sha256,
      rawSource: review?.plain_english, repairs: review?.adjudication?.patches || [],
      status: review ? review.adjudication.decision.replaceAll('_', ' ') + '; candidate' : d.status,
    });
  });
}

function loadSplit(root, catalog, split) {
  if (!['training', 'development', 'test'].includes(split)) throw Error('Unknown split');
  const meta = metadata(root, catalog), merged = new Map(), errors = [];
  for (const d of catalog.datasets.filter(d => d.split === split)) {
    try {
      for (const pair of loadDataset(root, d, meta)) {
        if (merged.has(pair.key)) {
          const prior = merged.get(pair.key);
          prior.datasets = [...new Set([...prior.datasets, ...pair.datasets])];
        } else merged.set(pair.key, pair);
      }
    } catch (error) { errors.push({ dataset: d.label || d.id, error: error.message }); }
  }
  return { pairs: [...merged.values()], errors };
}

function browserApp() {
  const $ = id => document.getElementById(id);
  const state = { pairs: [], filtered: [], index: 0 };
  const create = (tag, text, className) => {
    const el = document.createElement(tag); if (text !== undefined) el.textContent = text;
    if (className) el.className = className; return el;
  };
  const title = p => `${p.year ? p.year + ' · ' : ''}${p.title}`;
  function options(select, values, label) {
    const previous = select.value; select.replaceChildren(create('option', label)); select.firstChild.value = '';
    for (const [key, text] of values) { const el = create('option', text); el.value = key; select.append(el); }
    if ([...select.options].some(o => o.value === previous)) select.value = previous;
  }
  async function load() {
    $('status').textContent = 'Loading local data…';
    try {
      const response = await fetch('/api/pairs?split=' + encodeURIComponent($('split').value));
      const data = await response.json(); if (!response.ok) throw Error(data.error);
      state.pairs = data.pairs; state.index = 0;
      const sorted = set => [...set].sort((a,b) => a.localeCompare(b));
      options($('group'), sorted(new Set(state.pairs.map(p => p.group))).map(v => [v,v.replaceAll('_',' ')]), 'All author groups');
      options($('author'), sorted(new Set(state.pairs.flatMap(p => p.authors))).map(v => [v,v]), 'All individual authors');
      options($('teacher'), sorted(new Set(state.pairs.map(p => p.teacher))).map(v => [v,v]), 'Both / all teachers');
      $('warnings').textContent = data.errors.map(e => e.dataset + ': ' + e.error).join('\n');
      $('testNote').hidden = $('split').value !== 'test';
      $('testNote').textContent = 'You selected the reserved test split. Keep feedback from this material out of model selection. ' +
        (state.pairs.some(p => p.teacher === 'Gemini') ? '' : 'No Gemini counterpart is available in the loaded test catalog.');
      filter(true);
    } catch (error) { $('status').textContent = 'Could not load data: ' + error.message; }
  }
  function filter(refreshPapers = false) {
    let rows = state.pairs.filter(p => (!$('group').value || p.group === $('group').value) && (!$('author').value || p.authors.includes($('author').value)) && (!$('teacher').value || p.teacher === $('teacher').value));
    if (refreshPapers) {
      const papers = new Map(rows.map(p => [p.paperKey, title(p)]));
      options($('paper'), [...papers].sort((a,b) => a[1].localeCompare(b[1])), 'All papers');
    }
    const query = $('search').value.trim().toLocaleLowerCase();
    rows = rows.filter(p => (!$('paper').value || p.paperKey === $('paper').value) &&
      (!$('issue').checked || p.targetIssue || p.pending));
    const grouped = new Map();
    for (const pair of rows) {
      const key = pair.paperKey + ':' + pair.targetSha;
      if (!grouped.has(key)) grouped.set(key, { key, first: pair, variants: [] });
      grouped.get(key).variants.push(pair);
    }
    state.filtered = [...grouped.values()].filter(g => !query || g.variants.some(p =>
      [p.id,p.originalId,p.feedbackId,p.source,p.target,p.title,p.paperId,...p.authors].join(' ').toLocaleLowerCase().includes(query)));
    state.index = 0;
    $('status').textContent = `${state.filtered.length} human paragraphs · ${state.filtered.reduce((n,g)=>n+g.variants.length,0)} teacher-input variants`;
    const list = $('list'); list.replaceChildren();
    state.filtered.forEach((g, i) => {
      const button = create('button', undefined, 'list-item');
      button.append(create('strong', g.variants.map(p => p.id).join(' / ')), create('span', g.first.group.replaceAll('_',' ')), create('small', title(g.first)));
      button.onclick = () => { state.index = i; render(); }; list.append(button);
    });
    render();
  }
  function render() {
    const item = state.filtered[state.index]; $('detail').replaceChildren();
    $('previous').disabled = !item || state.index === 0; $('next').disabled = !item || state.index === state.filtered.length - 1;
    $('position').textContent = item ? `${state.index + 1} / ${state.filtered.length}` : 'No matches';
    [...$('list').children].forEach((el,i) => el.classList.toggle('active',i === state.index));
    if (!item) { $('detail').append(create('p','No matching pairs. Try another filter.')); return; }
    const p = item.first;
    $('detail').append(create('h2', title(p)), create('p', p.authors.join(' · '), 'authors'));
    if (p.targetIssue) $('detail').append(create('p', p.targetIssue, 'issue-note'));
    const target = create('section', undefined, 'target');
    target.append(create('h3','Original human paragraph · training target'),create('p',p.target,'prose')); $('detail').append(target);
    const variants = create('div',undefined,'variants');
    for (const v of item.variants.sort((a,b) => a.teacher.localeCompare(b.teacher))) {
      const card = create('section',undefined,'variant');
      card.append(create('h3',v.teacher + ' · ' + v.id),create('p',v.model,'model'),create('p',v.status,v.pending ? 'badge pending' : 'badge'));
      card.append(create('h4','Neutral prose · model input'),create('p',v.source,'prose'));
      card.append(create('p',v.feedbackId,'feedback-id'));
      if (v.rawSource !== v.source) {
        const raw = create('details'); raw.append(create('summary','Original teacher response and local repairs'),create('p',v.rawSource,'prose'));
        for (const patch of v.repairs) raw.append(create('p',patch.reason + '\n“' + patch.before + '” → “' + patch.after + '”','patch'));
        card.append(raw);
      }
      const provenance = create('details'); provenance.append(create('summary','Record details'));
      provenance.append(create('pre',JSON.stringify({ example_id:v.originalId, pair_id:v.feedbackId, paper_id:v.paperId, split:v.split, datasets:v.datasets, source_file:v.sourceFile, source_sha256:v.sourceSha, target_sha256:v.targetSha, page_location:v.sourceLocation },null,2)));
      card.append(provenance); variants.append(card);
    }
    $('detail').append(variants);
  }
  $('split').onchange = load; $('refresh').onclick = load;
  for (const id of ['group','author','teacher']) $(id).onchange = () => filter(true);
  $('paper').onchange = () => filter(); $('issue').onchange = () => filter();
  $('search').oninput = () => filter();
  $('previous').onclick = () => { state.index--; render(); }; $('next').onclick = () => { state.index++; render(); };
  document.addEventListener('keydown', e => {
    if (['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)) return;
    if (e.key === 'ArrowRight' && !$('next').disabled) $('next').click();
    if (e.key === 'ArrowLeft' && !$('previous').disabled) $('previous').click();
  });
  fetch('/api/catalog').then(r => r.json()).then(data => {
    $('catalogNote').textContent = data.description;
    for (const d of data.datasets) $('inventory').append(create('li',`${d.label} — ${d.split} · ${d.teacher || 'teacher by condition'}${d.declaredCount ? ' · ' + d.declaredCount + ' records (catalog)' : ''}`));
  }).catch(() => {});
  load();
}

function html() {
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pair review · local corpus</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#182638;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{padding:22px 28px;background:#172b3c;color:#fff}h1{font-size:24px;margin:0 0 5px}header p{margin:3px 0;color:#c5d7e3}main{padding:20px 28px}label{display:flex;flex-direction:column;font-size:12px;font-weight:650;gap:4px}select,input,button{font:inherit;color:inherit;border:1px solid #cbd4dd;border-radius:7px;background:#fff;padding:9px}button{cursor:pointer}button:disabled{opacity:.45;cursor:default}.filters{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.search{grid-column:span 2}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:15px 0}.controls label{display:flex;flex-direction:row;align-items:center;font-size:14px;font-weight:400}.layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:20px}.list{max-height:74vh;overflow:auto}.list-item{width:100%;text-align:left;display:flex;flex-direction:column;gap:3px;margin-bottom:7px;overflow-wrap:anywhere}.list-item small{line-height:1.35;color:#5b6878}.list-item.active{border-color:#267183;box-shadow:inset 3px 0 #267183;background:#eaf3f5}.list-item strong{font-size:13px}.list-item span{font-size:12px}h2{font-size:21px;line-height:1.4;margin:0 0 4px}h3{font-size:15px;margin:0 0 10px}h4{font-size:12px;color:#5b6878;margin:15px 0 5px}.authors,.model{color:#5b6878;margin:4px 0 12px}.target,.variant{border:1px solid #dbe1e7;border-radius:10px;padding:20px;background:#fff;margin:15px 0}.target{background:#fffaf0;border-color:#e8dcc3}.variants{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.variant{margin-top:0;min-width:0}.prose{font-family:Georgia,"Times New Roman",serif;font-size:17px;line-height:1.75;white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0}.badge{display:inline-block;background:#e9f2f1;color:#215b52;font-size:12px;padding:3px 8px;border-radius:5px}.pending,.issue-note{background:#fff0d8;color:#785017}.issue-note{padding:12px;border-radius:7px}.feedback-id{font-family:monospace;font-size:12px;color:#687887}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}details{margin:12px 0}summary{cursor:pointer;color:#486578;font-size:13px}.patch{font-size:12px;white-space:pre-wrap;border-left:2px solid #bdd2dd;padding-left:9px}#warnings{white-space:pre-wrap;color:#9c3131}#status{font-size:13px;color:#526577}.help{font-size:13px;color:#536476}.test-note{padding:12px;background:#fff0d8;border-radius:7px}footer{margin-top:25px}a{color:#216c83}@media(max-width:950px){.layout{grid-template-columns:1fr}.list{max-height:180px}.filters{grid-template-columns:repeat(2,minmax(0,1fr))}.search{grid-column:span 1}main{padding:14px}.variants{grid-template-columns:1fr}}:focus-visible{outline:3px solid #49a0b5;outline-offset:2px}
</style><header><h1>Academic prose · pair review</h1><p>Original human paragraph → teacher rewrite. For fine-tuning, the teacher rewrite is the input and the human paragraph is the target.</p></header>
<main><div class="filters">
<label>Split<select id="split"><option value="training">Training</option><option value="development">Development / validation</option><option value="test">Test · reserved historical set</option></select></label>
<label>Author group<select id="group"></select></label><label>Individual author<select id="author"></select></label>
<label>Paper<select id="paper"></select></label><label>Teacher / LLM<select id="teacher"></select></label>
<label>Search text or pair ID<input id="search" type="search" placeholder="P340, author name, term…"></label></div>
<div class="controls"><button id="refresh">Reload local data</button><label><input id="issue" type="checkbox">Target issues or pending input reviews only</label><span id="status" role="status"></span></div>
<p class="help">Read-only. Tell Codex your comments here using P/G IDs or the displayed pair ID. Identical copies are merged; edited versions remain distinct.</p>
<p id="testNote" class="test-note" hidden>Reserved test examples are shown only because you selected Test. Review may expose held-out material; keep resulting decisions out of model selection. A Gemini test counterpart has not been collected.</p>
<p id="warnings" role="alert"></p><div class="controls"><button id="previous">← Previous</button><span id="position"></span><button id="next">Next →</button></div>
<div class="layout"><nav id="list" class="list" aria-label="Human paragraphs"></nav><article id="detail"></article></div>
<footer><details><summary>Available datasets and coverage</summary><p id="catalogNote"></p><ul id="inventory"></ul><p class="help">Catalog changes are read on reload. Records marked candidate, staging or pending are not final training releases.</p></details></footer></main>
<script>(${browserApp.toString()})();</script></html>`;
}

function createServer(root = ROOT, catalogName = CATALOG) {
  return http.createServer((req, res) => {
    const allowed = /^127\.0\.0\.1(?::\d+)?$|^localhost(?::\d+)?$/;
    if (!allowed.test(req.headers.host || '') || (req.headers.origin && req.headers.origin !== 'http://' + req.headers.host)) { res.writeHead(403).end('Local access only'); return; }
    if (!['GET','HEAD'].includes(req.method)) { res.writeHead(405, { Allow:'GET, HEAD' }).end(); return; }
    res.setHeader('Cache-Control','no-store'); res.setHeader('X-Content-Type-Options','nosniff');
    try {
      const url = new URL(req.url, 'http://127.0.0.1');
      if (url.pathname === '/') { res.setHeader('Content-Type','text/html; charset=utf-8'); res.end(req.method === 'HEAD' ? '' : html()); return; }
      if (url.pathname === '/favicon.ico') { res.writeHead(204).end(); return; }
      res.setHeader('Content-Type','application/json; charset=utf-8');
      const catalog = readCatalog(root, catalogName);
      if (url.pathname === '/api/catalog') {
        res.end(JSON.stringify({ version:catalog.version, description:catalog.description, datasets:catalog.datasets.map(({id,label,split,teacher,declaredCount}) => ({id,label,split,teacher,declaredCount})) })); return;
      }
      if (url.pathname === '/api/pairs') { res.end(JSON.stringify(loadSplit(root, catalog, url.searchParams.get('split') || 'training'))); return; }
      res.writeHead(404).end(JSON.stringify({error:'Not found'}));
    } catch (error) { res.writeHead(400, {'Content-Type':'application/json; charset=utf-8'}).end(JSON.stringify({error:error.message})); }
  });
}

if (require.main === module) {
  const args = process.argv.slice(2);
  const arg = (name, fallback) => args.includes(name) ? args[args.indexOf(name)+1] : fallback;
  const catalogName = arg('--catalog', CATALOG);
  if (args.includes('--init-catalog')) {
    const file = localPath(ROOT, catalogName); fs.mkdirSync(path.dirname(file), {recursive:true});
    fs.writeFileSync(file, JSON.stringify(defaultCatalog(),null,2)+'\n', {flag:'wx'});
    console.log('Created local catalog: ' + file);
  } else if (args.includes('--validate')) {
    const catalog = readCatalog(ROOT,catalogName);
    for (const split of ['training','development']) {
      const result = loadSplit(ROOT,catalog,split);
      console.log(JSON.stringify({split, inputVariants:result.pairs.length, humanParagraphs:new Set(result.pairs.map(p=>p.paperKey+':'+p.targetSha)).size, teachers: [...new Set(result.pairs.map(p=>p.teacher))], errors:result.errors}));
      if (result.errors.length) process.exitCode = 1;
    }
    console.log('Reserved test text was not loaded.');
  } else {
    const port = Number(arg('--port', '8765')); if (!Number.isInteger(port) || port < 1 || port > 65535) throw Error('Invalid port');
    const server = createServer(ROOT,catalogName);
    server.on('error', error => { console.error('Local viewer could not start: ' + error.code); process.exitCode=1; });
    server.listen(port,'127.0.0.1',()=>console.log(`Local pair review: http://127.0.0.1:${port}\nRead-only; no provider calls. Reserved test text loads only if selected.\nCatalog: ${catalogName}`));
  }
}
module.exports = {sha, localPath, defaultCatalog, readCatalog, metadata, normalizePair, loadDataset, loadSplit, createServer, html};
