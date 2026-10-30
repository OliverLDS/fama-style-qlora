"""Small CPU-only, topic-pair-held-out authorship diagnostics using stdlib.

Fixed rhetorical cue proxies are not a syntactic parser or the LLM taxonomy.
No GPU, language model, sklearn, numpy or downloaded model is required.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics

from . import author_comparison_corpus as corpus
from fama_style.rhetoric.experiment import fresh, write_json, write_lines

PROTOCOL='author_comparison_diagnostics_v1'
CUES={
 'causal_connectives':r'\b(?:because|therefore|thus|hence|consequently|as a result)\b',
 'contrast_connectives':r'\b(?:although|however|but|nevertheless|in contrast|on the other hand)\b',
 'conditional_markers':r'\b(?:if|unless|provided that|as long as)\b',
 'uncertainty_markers':r'\b(?:may|might|could|perhaps|likely|unlikely|apparently|seems?|tends?)\b',
 'necessity_markers':r'\b(?:must|necessary|requires?|sufficient)\b',
 'definition_markers':r'\b(?:means|defined|definition|refer to|denote|call)\b',
 'first_person_plural':r'\b(?:we|our)\b',
 'first_person_singular':r'\b(?:i|my)\b',
 'evidence_references':r'\b(?:table|figure|appendix|section)\b',
 'enumeration_markers':r'\b(?:first|second|third|finally|respectively)\b',
 'attribution_markers':r'\b(?:argue|argues|suggest|suggests|report|reports|find|finds|show|shows)\b',
 'relative_clause_markers':r'\b(?:which|that|whose|whom)\b'}
RHETORICAL=list(CUES)+['initial_conditional_fraction','initial_connector_fraction','initial_evidence_fraction']
STRUCTURE=['semicolon_rate','colon_rate','parenthesis_rate','mean_word_length','nominalization_suffix_fraction','passive_ed_proxy']
NUISANCE=['word_count','sentences','mean_ocr_confidence','normalization_events','window_position','version_year','columns','author_count','unrecognized_word_fraction']
STOP=set('a an the of in on at to for from with by as is are was were be been being and or but if so that this these those it its they their we our i my he she his her not no can may might will would could should must do does did have has had than then there which who whom whose how what when where why into over under after before also more most less some any all both each such same other only very much many one two three about through between per during'.split())


def mask_metadata(text,names):
    text=re.sub(r'(?<=[A-Za-z][.!?])\d+(?=\s|$)','',text)
    text=re.sub(r'\([^)]*\b(?:18|19|20)\d{2}[^)]*\)',' ',text)
    for name in sorted(names,key=len,reverse=True):
        if len(name)>2:text=re.sub(r'\b'+re.escape(name)+r'\b',' ',text,flags=re.I)
    return text


def sentences(text):
    text=re.sub(r'\bU\.S\.','US',text)
    return [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])',text) if s.strip()]


def feature_values(row,names):
    text=mask_metadata(row['text'],names);ts=corpus.words(text);n=max(1,len(ts));ss=sentences(text);m=max(1,len(ss))
    f={name:100*len(re.findall(pattern,text,re.I))/n for name,pattern in CUES.items()}
    f.update(initial_conditional_fraction=sum(bool(re.match(r'\s*(?:if|unless|when|provided)\b',s,re.I)) for s in ss)/m,
             initial_connector_fraction=sum(bool(re.match(r'\s*(?:thus|therefore|however|hence|consequently|although)\b',s,re.I)) for s in ss)/m,
             initial_evidence_fraction=sum(bool(re.match(r'\s*(?:table|figure|appendix|section)\b',s,re.I)) for s in ss)/m,
             semicolon_rate=100*text.count(';')/n,colon_rate=100*text.count(':')/n,parenthesis_rate=100*text.count('(')/n,
             mean_word_length=statistics.mean(map(len,ts)) if ts else 0,
             nominalization_suffix_fraction=sum(bool(re.search(r'(?:tion|sion|ment|ness|ity)$',t.lower())) for t in ts)/n,
             passive_ed_proxy=100*len(re.findall(r'\b(?:is|are|was|were|be|been)\s+(?:\w+ly\s+)?\w+ed\b',text,re.I))/n,
             word_count=row['word_count'],sentences=m,mean_ocr_confidence=row['mean_ocr_confidence'],
             normalization_events=len(row['normalization_events']),window_position=row['window_position'],
             version_year=row['version_year'],columns=row['columns'],author_count=len(row['authors']),
             unrecognized_word_fraction=len(row['quality_flags']['unrecognized_lowercase_words'])/max(1,row['word_count']))
    return f


def folds(rows):
    for topic in sorted({r['topic'] for r in rows}):
        train=[i for i,r in enumerate(rows) if r['topic']!=topic];test=[i for i,r in enumerate(rows) if r['topic']==topic]
        if {rows[i]['paper_id'] for i in train}&{rows[i]['paper_id'] for i in test}:raise ValueError('paper_leakage')
        if {rows[i]['text_sha256'] for i in train}&{rows[i]['text_sha256'] for i in test}:raise ValueError('duplicate_text_leakage')
        if set(rows[i]['label'] for i in test)!={0,1}:raise ValueError('held_topic_must_contain_both_labels')
        yield topic,train,test


def sigmoid(x):return 1/(1+math.exp(-max(-30,min(30,x))))


def fit_ridge(x,y,weights,steps=350,penalty=.2):
    """Fixed L2 logistic fit; normalization estimated from training rows only."""
    n=len(x);d=len(x[0]);total=sum(weights)
    means=[sum(w*r[j] for w,r in zip(weights,x))/total for j in range(d)]
    scales=[max(.000001,math.sqrt(sum(w*(r[j]-means[j])**2 for w,r in zip(weights,x))/total)) for j in range(d)]
    z=[[max(-8,min(8,(v-means[j])/scales[j])) for j,v in enumerate(r)] for r in x]
    coef=[0.0]*d;intercept=0.0
    for _ in range(steps):
        errors=[(sigmoid(intercept+sum(a*b for a,b in zip(coef,r)))-label)*w/total for r,label,w in zip(z,y,weights)]
        gradient=[sum(e*r[j] for e,r in zip(errors,z))+penalty*coef[j] for j in range(d)]
        coef=[v-.25*g for v,g in zip(coef,gradient)];intercept-=.25*sum(errors)
    return {'means':means,'scales':scales,'coef':coef,'intercept':intercept}


def predict(model,row):
    z=[max(-8,min(8,(v-m)/s)) for v,m,s in zip(row,model['means'],model['scales'])]
    return sigmoid(model['intercept']+sum(a*b for a,b in zip(model['coef'],z)))


def content_tokens(row,names):
    text=mask_metadata(row['text'],names)
    text=re.sub(r'\b[A-Za-z]+-[A-Za-z]+\b',' ',text)  # avoid line-hyphen artifacts
    return [t.lower() for t in corpus.words(text) if len(t)>2 and t.lower() not in STOP]


def lexical_fit(rows,names):
    docs=[content_tokens(r,names) for r in rows];df=Counter(t for doc in docs for t in set(doc))
    vocab={t for t,c in df.items() if c>=2 and c<.85*len(rows)}
    counts=[Counter(),Counter()];papers=Counter(r['paper_id'] for r in rows)
    for r,doc in zip(rows,docs):
        for term,n in Counter(doc).items():
            if term in vocab:counts[r['label']][term]+=n/papers[r['paper_id']]
    totals=[sum(c.values())+max(1,len(vocab)) for c in counts]
    logodds={t:math.log((counts[1][t]+1)/totals[1])-math.log((counts[0][t]+1)/totals[0]) for t in vocab}
    return logodds


def lexical_predict(model,row,names):return sigmoid(sum(model.get(t,0)*n for t,n in Counter(content_tokens(row,names)).items()))


def run(corpus_dir,output):
    manifest=json.loads((corpus_dir/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if corpus.file_sha(corpus_dir/name)!=digest:raise ValueError('corpus_artifact_changed')
    if manifest['code_sha256']!=corpus.file_sha(Path(corpus.__file__)):raise ValueError('corpus_builder_changed')
    rows=corpus.read_lines(corpus_dir/'paragraphs.private.jsonl');registry=json.loads((corpus_dir/'registry.private.json').read_text())
    corpus.validate_registry(registry)
    authors={word.lower() for p in registry['papers'] for author in p['authors'] for word in corpus.words(author)}
    source_papers={p['paper_id']:p for p in registry['papers']}
    for row in rows:row['columns']=source_papers[row['paper_id']]['columns']
    features=[feature_values(r,authors) for r in rows]
    panels={'rhetorical_cues':RHETORICAL,'structure_without_length':STRUCTURE,
            'rhetorical_plus_structure':RHETORICAL+STRUCTURE,'length_acquisition_metadata':NUISANCE,'content_lexical':None}
    predictions=[];models=[]
    for topic,train,test in folds(rows):
        train_rows=[rows[i] for i in train];counts=Counter(r['paper_id'] for r in train_rows)
        weights=[1/counts[r['paper_id']] for r in train_rows]
        for panel,keys in panels.items():
            if keys:
                model=fit_ridge([[features[i][k] for k in keys] for i in train],[r['label'] for r in train_rows],weights)
                scores=[predict(model,[features[i][k] for k in keys]) for i in test]
                models.append({'topic_held_out':topic,'panel':panel,'feature_coefficients':dict(zip(keys,model['coef']))})
            else:
                model=lexical_fit(train_rows,authors);scores=[lexical_predict(model,rows[i],authors) for i in test]
                models.append({'topic_held_out':topic,'panel':panel,'vocabulary_size':len(model)})
            predictions += [{'id':rows[i]['id'],'paper_id':rows[i]['paper_id'],'topic':topic,'label':rows[i]['label'],'panel':panel,'probability':p} for i,p in zip(test,scores)]
    results={}
    for panel in panels:
        group=[r for r in predictions if r['panel']==panel];papers=[]
        for pid in sorted({r['paper_id'] for r in group}):
            rs=[r for r in group if r['paper_id']==pid];p=statistics.mean(r['probability'] for r in rs)
            papers.append({'paper_id':pid,'topic':rs[0]['topic'],'label':rs[0]['label'],'mean_probability':p,'correct':int(p>=.5)==rs[0]['label']})
        wins=sum(next(p['mean_probability'] for p in papers if p['topic']==t and p['label']==1)>next(p['mean_probability'] for p in papers if p['topic']==t and p['label']==0) for t in {p['topic'] for p in papers})
        results[panel]={'paper_accuracy':statistics.mean(p['correct'] for p in papers),'papers_correct':sum(p['correct'] for p in papers),'papers':len(papers),
            'topic_pair_rank_wins':wins,'topic_pairs':7,'paragraph_accuracy_descriptive_only':statistics.mean(int(p['probability']>=.5)==p['label'] for p in group),
            'per_paper':papers}
    associations=[]
    for name in RHETORICAL+STRUCTURE:
        differences=[];signs=[]
        for topic in sorted({r['topic'] for r in rows}):
            means={label:statistics.mean(features[i][name] for i,r in enumerate(rows) if r['topic']==topic and r['label']==label) for label in (0,1)}
            differences.append({'topic':topic,'target_minus_control':means[1]-means[0]})
        cs=[m['feature_coefficients'][name] for m in models if m['panel']=='rhetorical_plus_structure']
        positive=sum(d['target_minus_control']>1e-9 for d in differences);negative=sum(d['target_minus_control']< -1e-9 for d in differences)
        associations.append({'feature':name,'positive_topic_pairs':positive,'negative_topic_pairs':negative,'ties':7-positive-negative,
            'median_pair_difference':statistics.median(d['target_minus_control'] for d in differences),
            'positive_fold_coefficients':sum(v>1e-9 for v in cs),'negative_fold_coefficients':sum(v< -1e-9 for v in cs),
            'consistent_descriptive_candidate':(positive>=6 and sum(v>1e-9 for v in cs)>=6) or (negative>=6 and sum(v< -1e-9 for v in cs)>=6),
            'per_topic_differences':differences})
    summary={'protocol':PROTOCOL,'status':'executed_exploratory_cpu_diagnostic','paragraphs':len(rows),'papers':14,'topic_pairs':7,
        'evaluation':'Leave one entire matched topic/paper pair out; train-only scaling and vocabulary; paper-weighted fixed L2 logistic, lexical NB baseline.',
        'hyperparameters':{'logistic_steps':350,'L2_penalty':.2,'step_size':.25,'label_threshold':.5,'lexical_alpha':1},
        'primary_unit':'paper; paragraphs are not independent author samples','panels':results,'feature_associations':associations,
        'independent_taxonomy_extraction':False,'features_are_rhetorical_surface_proxies':True,'author_specificity_established':False,
        'causal_rewriting_benefit_established':False,'validation_rows_read':0,'reserved_test_rows_read':0,
        'language_model_calls':0,'gpu_jobs_started':0,'adapter_training_called':False,'cpu_classifier_fits':35}
    fresh(output);write_lines(output/'predictions.private.jsonl',predictions);write_lines(output/'features.private.jsonl',[{'id':r['id'],'paper_id':r['paper_id'],'features':f} for r,f in zip(rows,features)])
    write_json(output/'fold_models.private.json',models);write_json(output/'summary.json',summary)
    write_json(output/'manifest.json',{'protocol':PROTOCOL,'code_sha256':corpus.file_sha(Path(__file__)),
        'corpus_manifest_sha256':corpus.file_sha(corpus_dir/'manifest.json'),
        'files':{p.name:corpus.file_sha(p) for p in output.iterdir() if p.is_file()}})
    print(json.dumps({panel:{k:v[k] for k in ('papers_correct','papers','topic_pair_rank_wins')} for panel,v in results.items()}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--corpus-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();run(a.corpus_dir,a.output_dir)


if __name__=='__main__':main()
