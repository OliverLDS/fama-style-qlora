"""Select private training-only discovery material; never reads held-out splits."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SHA256 = 'a108416c1935615a7e01bb4d3fa86e0818b961bfb16f15d3883e6905654ee248'
PAPERS = ('F1965','F1980','FF2000','FJ1983','ME1997','NL2003','VL2018')
SEED = 20261020
CUES = {
    'condition': r'\b(if|unless|when|provided)\b',
    'comparison': r'\b(than|compared|relative|whereas)\b',
    'causal_expression': r'\b(because|causes?|therefore|hence|leads? to)\b',
    'uncertainty': r'\b(may|might|could|possible|expected)\b',
    'reporting': r'\b(data|estimates?|coefficients?|results?|table|we find)\b',
    'definition': r'\b(defined|denotes|refers to)\b',
    'qualification': r'\b(except|unless|however|although|but)\b',
    'negation': r'\b(not|no|never|without)\b',
}


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def words(text):
    return len(re.findall(r"\b[\w’'-]+\b", text))


def tags(row):
    text=row['completion']+' '+row['plain_english']
    return {key for key,pattern in CUES.items() if re.search(pattern,text,re.I)}


def select(rows):
    if any(row.get('split')!='train' for row in rows): raise ValueError('Training-only discovery required')
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate training IDs')
    groups=collections.defaultdict(list)
    for row in rows:
        if not row.get('plain_english','').strip() or not row.get('completion','').strip(): raise ValueError('Empty pair')
        groups[row['paper_key']].append(row)
    if set(groups)!=set(PAPERS) or any(len(group)<4 for group in groups.values()):
        raise ValueError('Need all seven training papers with at least four rows each')
    selected=[]
    for paper in PAPERS:
        ordered=sorted(groups[paper],key=lambda r:(words(r['completion']),r['id']))
        covered=set();previous=[];n=len(ordered)
        for quartile in range(4):
            group=ordered[quartile*n//4:(quartile+1)*n//4]
            middle=statistics.median(words(r['completion']) for r in group)
            def rank(row):
                observed=tags(row)
                distance=min((len(observed^before) for before in previous),default=0)
                return (-len(observed-covered),-distance,-len(observed),abs(words(row['completion'])-middle),row['id'])
            row=min(group,key=rank);covered.update(tags(row));previous.append(tags(row))
            selected.append({'row':row,'length_quartile':quartile+1,'sampling_cues':sorted(tags(row))})
    return selected


def lines(path, values):
    path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values),encoding='utf-8')


def js(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def build(rows, output):
    if output.exists() and any(output.iterdir()): raise ValueError('Refusing to overwrite a discovery package')
    chosen=select(rows)
    output.mkdir(parents=True,exist_ok=True)
    materials={name:[] for name in ('A','B','pairs')};key=[]
    for item in chosen:
        row=item['row']
        def opaque(kind):return digest(f'{SEED}|{kind}|{row["id"]}')[:16]
        a='p_'+opaque('original');b='p_'+opaque('reference');pair='pair_'+opaque('pair');proposal='n_'+opaque('proposal')
        materials['A'].append({'paragraph_id':a,'text':row['completion']})
        materials['B'].append({'paragraph_id':b,'text':row['plain_english']})
        materials['pairs'].append({'pair_id':pair,'original_item_id':a,'reference_id':b,'proposal_id':proposal,
                                   'original_paragraph':row['completion'],'plain_english_reference':row['plain_english']})
        key.append({'training_id':row['id'],'paper_key':row['paper_key'],'split':'train','original_item_id':a,
                    'reference_id':b,'pair_id':pair,'proposal_id':proposal,'length_quartile':item['length_quartile'],
                    'sampling_cues':item['sampling_cues'],'original_words':words(row['completion']),
                    'reference_words':words(row['plain_english']),'original_sha256':digest(row['completion']),
                    'reference_sha256':digest(row['plain_english'])})
    # Independently shuffle lists: row order and paragraph IDs do not reveal pair links.
    for index, name in enumerate(materials):random.Random(SEED+index).shuffle(materials[name])
    for name,stem in (('A','paragraphs_A'),('B','paragraphs_B'),('pairs','neutral_reference_pairs')):
        lines(output/(stem+'.jsonl'),materials[name])
        for index in range(4):
            lines(output/(stem+f'_{index+1:02}.jsonl'),materials[name][index*7:(index+1)*7])
    assets=ROOT/'docs/taxonomy_discovery_v1'
    for path in sorted(assets.iterdir()):
        if path.is_file():shutil.copyfile(path,output/path.name)
    js(output/'PRIVATE_MAPPING_DO_NOT_UPLOAD.json',key)
    report={'training_pairs':len(rows),'selected_pairs':len(chosen),'selected_paragraphs':2*len(chosen),
        'papers':dict(collections.Counter(item['paper_key'] for item in key)),
        'selection':'One row per within-paper original-length quartile, favoring new surface cue coverage; deterministic ties',
        'seed':SEED,'sampling_cues_are_not_annotations':True,'representative_random_sample':False,
        'validation_rows_read':0,'reserved_test_rows_read':0,'reference_text_modified':False,
        'batch_B_role':'observed reference only; neutral-design task may propose new text and taxonomy separately',
        'limits':'Deliberately diversified sample; cue heuristics are not verified reasoning labels or population estimates'}
    js(output/'selection_report.json',report)
    js(output/'manifest.json',{'implementation':'taxonomy_discovery_pack_v1','training_sha256':TRAIN_SHA256,
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    path=ROOT/'data/fama_corpus_v1/train.jsonl';raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=TRAIN_SHA256:raise ValueError('Frozen training file hash mismatch')
    rows=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(rows)!=247:raise ValueError('Expected 247 frozen training pairs')
    report=build(rows,args.output_dir)
    print(json.dumps({'selected_pairs':report['selected_pairs'],'selected_paragraphs':report['selected_paragraphs'],
                      'output_dir':str(args.output_dir),'test_rows_read':0,'training_called':False}))


if __name__=='__main__':main()
