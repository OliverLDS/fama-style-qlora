'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const vm = require('node:vm');
const http = require('node:http');
const review = require('../scripts/review_pairs.js');

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(),'pair-review-'));
  t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
  const write = (file, data) => { fs.mkdirSync(path.dirname(path.join(root,file)),{recursive:true}); fs.writeFileSync(path.join(root,file),data); };
  const line = row => JSON.stringify(row)+'\n';
  const row = {id:'stable-1',paper_id:'paper-1',authors:['Author One'],source:'Neutral text.',target:'Human paragraph.'};
  const dataset = {id:'future-release',label:'Future release',adapter:'pairs',split:'training',teacher:'Future LLM',model:'future-version',path:'data/pairs.jsonl'};
  write(dataset.path,line(row));
  const catalog = {version:1,datasets:[dataset]};
  return {root,write,line,row,dataset,catalog};
}

test('new generic releases load through the catalog without viewer code changes',t=>{
  const f=fixture(t),result=review.loadSplit(f.root,f.catalog,'training');
  assert.equal(result.errors.length,0);assert.equal(result.pairs.length,1);
  assert.equal(result.pairs[0].teacher,'Future LLM');assert.equal(result.pairs[0].model,'future-version');
});

test('identical copies merge provenance while edited versions remain distinct',t=>{
  const f=fixture(t);
  f.catalog.datasets.push({...f.dataset,id:'duplicate',label:'Another release'});
  let result=review.loadSplit(f.root,f.catalog,'training');
  assert.equal(result.pairs.length,1);assert.deepEqual(result.pairs[0].datasets,['Future release','Another release']);
  f.write('data/revised.jsonl',f.line({...f.row,source:'A revised neutral input.'}));
  f.catalog.datasets.push({...f.dataset,id:'revision',path:'data/revised.jsonl'});
  result=review.loadSplit(f.root,f.catalog,'training');assert.equal(result.pairs.length,2);
  assert.notEqual(result.pairs[0].feedbackId,result.pairs[1].feedbackId);
  assert.equal(result.pairs[0].targetSha,result.pairs[1].targetSha);
});

test('feedback identity depends on text rather than row order or display labels',t=>{
  const f=fixture(t),meta=review.metadata(f.root,f.catalog);
  const a=review.normalizePair(f.row,f.dataset,meta);
  const b=review.normalizePair(f.row,{...f.dataset,id:'renamed',label:'New label'},meta);
  assert.equal(a.feedbackId,b.feedbackId);
  assert.throws(()=>review.normalizePair({...f.row,target_sha256:'tampered'},f.dataset,meta),/hash mismatch/);
});

test('training and development loads never read test bodies',t=>{
  const f=fixture(t);
  f.write('data/reserved.jsonl','INVALID: this must not be parsed during training');
  f.catalog.datasets.push({...f.dataset,id:'reserved',split:'test',path:'data/reserved.jsonl'});
  assert.equal(review.loadSplit(f.root,f.catalog,'training').errors.length,0);
  assert.equal(review.loadSplit(f.root,f.catalog,'development').errors.length,0);
  assert.equal(review.loadSplit(f.root,f.catalog,'test').errors.length,1);
});

test('alternate JSONL field names are catalog-configurable',t=>{
  const f=fixture(t);f.write('data/custom.jsonl',f.line({id:'new',input_text:'Input.',reference_text:'Target.'}));
  f.catalog.datasets=[{...f.dataset,path:'data/custom.jsonl',sourceFields:['input_text'],targetFields:['reference_text']}];
  const result=review.loadSplit(f.root,f.catalog,'training');assert.equal(result.errors.length,0);assert.equal(result.pairs[0].source,'Input.');
});

test('catalog paths cannot escape the project or follow external symlinks',t=>{
  const f=fixture(t);assert.throws(()=>review.localPath(f.root,'../outside.jsonl'),/inside/);
  fs.symlinkSync(os.tmpdir(),path.join(f.root,'external'));assert.throws(()=>review.localPath(f.root,'external'),/symlinks/);
});

test('browser client parses and inserts prose as text, without HTML interpolation',()=>{
  const html=review.html();const script=html.match(/<script>([\s\S]*)<\/script>/)[1];
  new vm.Script(script);
  assert.ok(script.includes('textContent'));
  assert.ok(!script.includes('innerHTML'));
  assert.ok(!html.includes('Export feedback'));
});

test('HTTP service is read-only, local-origin only, and exposes catalog data',async t=>{
  const f=fixture(t);f.write('catalog.json',JSON.stringify(f.catalog));
  const server=review.createServer(f.root,'catalog.json');
  await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(0,'127.0.0.1',resolve);});
  t.after(()=>new Promise(resolve=>server.close(resolve)));
  const base=`http://127.0.0.1:${server.address().port}`;
  let response=await fetch(base+'/api/pairs?split=training');assert.equal(response.status,200);
  assert.equal((await response.json()).pairs[0].source,'Neutral text.');
  response=await fetch(base+'/api/pairs',{method:'POST',body:'overwrite'});assert.equal(response.status,405);
  response=await fetch(base+'/api/pairs',{headers:{Origin:'https://example.com'}});assert.equal(response.status,403);
  const hostStatus=await new Promise((resolve,reject)=>http.get(base+'/api/pairs',{headers:{Host:'example.com'}},r=>{r.resume();resolve(r.statusCode);}).on('error',reject));
  assert.equal(hostStatus,403);
  response=await fetch(base+'/data/pairs.jsonl');assert.equal(response.status,404);
  assert.equal(fs.readFileSync(path.join(f.root,f.dataset.path),'utf8'),f.line(f.row));
});

const project=path.resolve(__dirname,'..');
test('actual catalog retains paired training, staged statuses and no duplicate targets',{skip:!fs.existsSync(path.join(project,'research_runs/pair_review_v1/catalog.private.json'))},()=>{
  const all=review.readCatalog(project);const catalog={...all,datasets:all.datasets.filter(d=>['gpt6-training-500','gemini-training-500'].includes(d.id))};
  const result=review.loadSplit(project,catalog,'training');
  assert.deepEqual(result.errors,[]);assert.equal(result.pairs.length,1000);
  assert.equal(new Set(result.pairs.map(p=>p.paperKey+':'+p.targetSha)).size,500);
  assert.equal(result.pairs.filter(p=>p.teacher==='Gemini'&&p.pending).length,46);
  for (const id of ['P250','P264','P291','P458','P484']) assert.ok(result.pairs.find(p=>p.id===id&&p.teacher==='GPT-6').targetIssue);
  assert.equal(result.pairs.filter(p=>p.teacher==='GPT-6'&&p.repaired).length,10);
  assert.equal(result.pairs.filter(p=>p.teacher==='Gemini'&&p.repaired).length,23);
});

test('new brief-reconstruction release loads through the generic catalog adapter',{skip:!fs.existsSync(path.join(project,'research_runs/data_scaling_v1/gpt6_track_0500_v1/contrast_rewrite_v1/rewrite_quality_v1/manifest.json'))},()=>{
  const all=review.readCatalog(project);const d=all.datasets.find(d=>d.id==='gpt6-brief-reconstruction-v1');
  assert.ok(d);const result=review.loadSplit(project,{...all,datasets:[d]},'training');
  assert.deepEqual(result.errors,[]);assert.equal(result.pairs.length,500);
  assert.equal(result.pairs.filter(p=>p.status==='Candidate · fresh brief reconstruction').length,173);
  assert.ok(result.pairs.find(p=>p.id==='P024').source.startsWith('Because brokerage records'));
  assert.ok(!/\bhope\b/i.test(result.pairs.find(p=>p.id==='P002').source));
});
