"""Generate self-contained notebooks for manual Kaggle UI upload; never submits a run."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]

SOURCE_FILES = {'track_reviewed_4b': 'src/fama_style/training/track_reviewed_4b.py', 'train': 'src/fama_style/training/train.py', 'smoke_4b_reviewed': 'src/fama_style/training/smoke_4b_reviewed.py', 'compare_few_shot': 'src/fama_style/evaluation/compare_few_shot.py', 'compare_style': 'src/fama_style/evaluation/compare_style.py', 'continue_overfit_two': 'src/fama_style/diagnostics/continue_overfit_two.py', 'audit_training_labels': 'src/fama_style/diagnostics/audit_training_labels.py', 'diagnose_copying': 'src/fama_style/diagnostics/diagnose_copying.py', 'overfit_two': 'src/fama_style/diagnostics/overfit_two.py', 'audit_label_preprocessing': 'src/fama_style/diagnostics/audit_label_preprocessing.py', 'check_epoch_adapters': 'src/fama_style/diagnostics/check_epoch_adapters.py', 'track_validation_epochs': 'src/fama_style/diagnostics/track_validation_epochs.py'}
SOURCE_FILES['corpus_baseline'] = 'src/fama_style/evaluation/corpus_baseline.py'
SOURCE_FILES['corpus_smoke'] = 'src/fama_style/training/corpus_smoke.py'
SOURCE_FILES['corpus_pilot'] = 'src/fama_style/training/corpus_pilot.py'

def notebook_source(name):
    """Read package code and make its local imports work without a repo checkout."""
    text = (root / SOURCE_FILES[name]).read_text(encoding="utf-8")
    prompts = {"compare_style": "rewrite", "compare_few_shot": "few_shot", "corpus_baseline": "corpus"}
    if name in prompts:
        prompt_name = prompts[name]
        prompt_text = (root / "src/fama_style/prompting" / (prompt_name + ".py")).read_text(encoding="utf-8")
        text = re.sub(r"(?m)^from fama_style\.prompting\.[a-z_]+ import [^\n]+\n", "", text)
        text = prompt_text + "\n\n" + text
    text = re.sub(r"(?m)^from fama_style\.[a-z_]+\.([a-z_0-9]+) import ", r"from \1 import ", text)
    text = text.replace('from fama_style.evaluation import corpus_baseline as baseline',
                        'import corpus_baseline as baseline')
    text = text.replace('from fama_style.training import corpus_smoke as smoke',
                        'import corpus_smoke as smoke')
    return text

script = notebook_source('train')
script = script[:script.rfind('\nif __name__ == "__main__":')]


def cell(source):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[], 'source':[line+'\n' for line in source.rstrip().split('\n')]}

book = {
 'cells':[
   {'cell_type':'markdown','metadata':{},'source':['# Gemma 3 1B instructed Fama-Jensen QLoRA pilot\n', 'Attach the private Kaggle Dataset containing `fama_jensen_1983_pilot_instructed.jsonl`, enable the `HF_TOKEN` Secret, and select a T4 GPU with internet access. Every prompt asks for a meaning-preserving rewrite. This notebook runs one pilot epoch on 54 training pairs, checks 12 validation pairs using the same prompt format, and leaves 12 test pairs untouched. It fails before training if the instructed dataset is missing or ambiguous.\n']},
   cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"'),
   cell('%pip install -q -U "transformers>=4.50,<6" "trl>=0.20" "peft>=0.15" "accelerate>=1.0" "datasets>=3.0" "bitsandbytes>=0.46"'),
   cell(script + '\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nif len(files) != 1:\n    raise FileNotFoundError(f"Expected one attached instructed pilot JSONL under /kaggle/input; found {len(files)}")\nrows = load_rows(files[0])\nif any(row.get("prompt_format") != "rewrite_v1" or not row["prompt"].startswith("Rewrite the passage below") for row in rows):\n    raise ValueError("Attached dataset does not contain instructed rewrite_v1 prompts")\nos.environ.pop("MAX_STEPS", None)\nmain(["--method", "qlora", "--data", str(files[0]), "--epochs", "1"])'),
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
 'nbformat':4,'nbformat_minor':5
}
(root / 'kaggle' / 'train.ipynb').write_text(json.dumps(book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

comparison_script = notebook_source('compare_style')
comparison_script = comparison_script[:comparison_script.rfind('\nif __name__ == "__main__":')]
comparison_book = {
 'cells':[
   {'cell_type':'markdown','metadata':{},'source':['# Instructed Fama-Jensen style comparison\n', 'Attach the private instructed 78-pair pilot Dataset and the completed instructed-training notebook output through **Add Input**. Enable `HF_TOKEN`, select a T4 GPU, and allow internet access. This notebook does inference only on the 12 untouched test pairs using the exact training prompt format. It saves side-by-side generations and a manual claim-review worksheet under `/kaggle/working`.\n']},
   cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"'),
   cell('%pip install -q -U "transformers>=4.50,<6" "peft>=0.15" "accelerate>=1.0" "bitsandbytes>=0.46"'),
   cell(comparison_script + '\n\ndatasets = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nadapters = list(Path("/kaggle/input").rglob("gemma3_adapter/adapter_config.json"))\nif len(datasets) != 1 or len(adapters) != 1:\n    raise FileNotFoundError(f"Expected one instructed pilot dataset and one instructed-pilot adapter input; found {len(datasets)} datasets and {len(adapters)} adapters")\nmain(["--dataset", str(datasets[0]), "--adapter", str(adapters[0].parent), "--require-rewrite", "--output", "/kaggle/working/fama_style_comparison_instructed.jsonl"])'),
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
 'nbformat':4,'nbformat_minor':5
}
(root / 'kaggle' / 'compare_style.ipynb').write_text(json.dumps(comparison_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

diagnostic_book = {
 'cells':[
   {'cell_type':'markdown','metadata':{},'source':['# Two-example rewrite prompt diagnostic\n', 'Attach the private 78-pair pilot Dataset and version 7 training notebook output through **Add Input**. Enable `HF_TOKEN`, select T4, and allow internet. This notebook does **inference only** on held-out `p003` and `p043`, asking base and adapted Gemma to rewrite each passage. It saves `/kaggle/working/fama_prompt_diagnostic.jsonl`. The adapter was trained without this instruction, so this is a prompt diagnostic, not a fair evaluation of instruction-trained fine-tuning.\n']},
   cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"'),
   cell('%pip install -q -U "transformers>=4.50,<6" "peft>=0.15" "accelerate>=1.0" "bitsandbytes>=0.46"'),
   cell(comparison_script + '\n\ndatasets = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot.jsonl"))\nadapters = list(Path("/kaggle/input").rglob("gemma3_adapter/adapter_config.json"))\nif len(datasets) != 1 or len(adapters) != 1:\n    raise FileNotFoundError(f"Expected one pilot dataset and one version 7 adapter input; found {len(datasets)} datasets and {len(adapters)} adapters")\nmain(["--dataset", str(datasets[0]), "--adapter", str(adapters[0].parent), "--source-ids", "p003,p043", "--instruction", "--max-new-tokens", "256", "--output", "/kaggle/working/fama_prompt_diagnostic.jsonl"])'),
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
 'nbformat':4,'nbformat_minor':5
}
(root / 'kaggle' / 'prompt_diagnostic.ipynb').write_text(json.dumps(diagnostic_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

copy_script = notebook_source('diagnose_copying')
copy_script = copy_script[:copy_script.rfind('\nif __name__ == "__main__":')]
copy_book = {
 'cells':[
   {'cell_type':'markdown','metadata':{},'source':['# Instructed pilot copying diagnostic\n', 'Attach the private instructed 78-pair Dataset and **training version 9** output through **Add Input**. Enable `HF_TOKEN`, select T4, and allow internet access. This notebook performs inference only: two seen training pairs (`p001`, `p014`) and two untouched test pairs (`p003`, `p045`). It compares base and adapted greedy outputs, tries one sampled adapted output per pair, and audits all 78 tokenized prompt/target lengths. It saves `copy_diagnostic.jsonl` and `copy_token_audit.json` under `/kaggle/working`. It does not retrain or modify the adapter.\n']},
   cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"'),
   cell('%pip install -q -U "transformers>=4.50,<6" "peft>=0.15" "accelerate>=1.0" "bitsandbytes>=0.46"'),
   cell(comparison_script + '\n\n' + copy_script.replace('from compare_style import MODEL_ID, generation_prompt, load_hf_token\n', '') + '\n\ndatasets = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nadapters = list(Path("/kaggle/input").rglob("gemma3_adapter/adapter_config.json"))\nif len(datasets) != 1 or len(adapters) != 1:\n    raise FileNotFoundError(f"Expected one instructed dataset and one version 9 adapter input; found {len(datasets)} datasets and {len(adapters)} adapters")\nmain(["--dataset", str(datasets[0]), "--adapter", str(adapters[0].parent), "--output-dir", "/kaggle/working"])'),
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
 'nbformat':4,'nbformat_minor':5
}
(root / 'kaggle' / 'copy_diagnostic.ipynb').write_text(json.dumps(copy_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

label_script = notebook_source('audit_training_labels')
label_script = label_script[:label_script.rfind('\nif __name__ == "__main__":')]
label_book = {
 'cells':[
   {'cell_type':'markdown','metadata':{},'source':['# Gemma pilot supervised-label audit\n', 'Attach only the private instructed 78-pair pilot Dataset, enable `HF_TOKEN`, select a T4 GPU, and allow internet access. This notebook reconstructs the version 9 QLoRA `SFTTrainer` and inspects its prepared train/validation batches. It does **not** call `trainer.train()` or save an adapter. It writes `/kaggle/working/label_audit.json` with counts and match flags only; source passages and Secret values are not written to the report.\n']},
   cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"'),
   cell('%pip install -q -U "transformers>=4.50,<6" "trl>=0.20" "peft>=0.15" "accelerate>=1.0" "datasets>=3.0" "bitsandbytes>=0.46"'),
   cell(label_script + '\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nif len(files) != 1:\n    raise FileNotFoundError(f"Expected one instructed pilot dataset; found {len(files)}")\nmain(["--dataset", str(files[0]), "--output", "/kaggle/working/label_audit.json"])'),
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
 'nbformat':4,'nbformat_minor':5
}
(root / 'kaggle' / 'label_audit.ipynb').write_text(json.dumps(label_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

fast_label_script = notebook_source('audit_label_preprocessing')
fast_label_script = fast_label_script[:fast_label_script.rfind('\nif __name__ == "__main__":')]
fast_label_book = {
 'cells':[
   {'cell_type':'markdown','metadata':{},'source':['# Fast Gemma label-preparation audit\n', 'Attach the private instructed 78-pair pilot Dataset and enable `HF_TOKEN`. A CPU is sufficient. This notebook loads the Gemma tokenizer only, reconstructs TRL-style completion masking and 1024-token truncation, and saves counts and match flags to `/kaggle/working/label_preprocessing_audit.json`. It does not download model weights, train, or expose the source passages in its report. The separate `label_audit.ipynb` constructs the full trainer if an exact prepared-batch inspection remains necessary.\n']},
   cell('%pip install -q -U "transformers>=4.50,<6"'),
   cell(fast_label_script + '\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nif len(files) != 1:\n    raise FileNotFoundError(f"Expected one instructed pilot dataset; found {len(files)}")\nmain(["--dataset", str(files[0]), "--output", "/kaggle/working/label_preprocessing_audit.json"])'),
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
 'nbformat':4,'nbformat_minor':5
}
(root / 'kaggle' / 'label_audit_fast.ipynb').write_text(json.dumps(fast_label_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

overfit_script = notebook_source('overfit_two')
overfit_script = overfit_script[:overfit_script.rfind('\nif __name__ == "__main__":')]
overfit_book = {
 'cells':[
   {'cell_type':'markdown','metadata':{},'source':['# Two-example Gemma QLoRA memorization diagnostic\n', 'Import as a separate **private** Kaggle notebook. Attach only the private instructed 78-pair Dataset containing `fama_jensen_1983_pilot_instructed.jsonl`. Enable the `HF_TOKEN` Secret, select a T4 GPU, and allow internet. This trains a **fresh** QLoRA adapter on only train rows `p001` and `p014` for 60 optimizer steps. It records train-pair losses and deterministic generations at steps 0, 10, 30, and 60; untouched test row `p003` is generated only at steps 0 and 60. Results are saved to `/kaggle/working/overfit_two/overfit_two_report.json`. This is a memorization diagnostic, not a style evaluation or full training run.\n']},
   cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"'),
   cell('%pip install -q -U "transformers>=4.50,<6" "trl>=0.20" "peft>=0.15" "accelerate>=1.0" "datasets>=3.0" "bitsandbytes>=0.46"'),
   cell(overfit_script + '\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nif len(files) != 1:\n    raise FileNotFoundError(f"Expected one instructed pilot dataset; found {len(files)}")\nmain(["--dataset", str(files[0]), "--output-dir", "/kaggle/working/overfit_two"])'),
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
 'nbformat':4,'nbformat_minor':5
}
(root / 'kaggle' / 'overfit_two.ipynb').write_text(json.dumps(overfit_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

continuation_script = notebook_source('continue_overfit_two')
continuation_script = continuation_script[:continuation_script.rfind('\nif __name__ == "__main__":')]
local_import = ('from overfit_two import (HELD_OUT_ID, MAX_LENGTH, MODEL_ID, TRAIN_IDS,\n'
                '                         load_examples, load_hf_token, snapshot, to_chat_rows)\n')
assert local_import in continuation_script
continuation_script = continuation_script.replace(local_import, '')
continuation_book = {
 'cells':[
   {'cell_type':'markdown','metadata':{},'source':['# Continue the two-example Gemma QLoRA diagnostic\n', 'Import as a separate **private** Kaggle notebook. Attach the private instructed 78-pair Dataset **and** the completed output from the two-example diagnostic (the run with `overfit_two/adapter/adapter_config.json` and `overfit_two/overfit_two_report.json`). Enable the `HF_TOKEN` Secret, select a T4 GPU, and allow internet. This loads the saved 60-step adapter and trains only `p001` and `p014` for 120 **additional** optimizer steps. It records losses and deterministic generations after 0, 30, 60, and 120 added steps, with untouched test row `p003` generated only at the start and end. The earlier run did not save optimizer state, so AdamW and the learning-rate schedule restart at the same settings. The output is `/kaggle/working/overfit_two_continued/overfit_two_continuation_report.json`.\n']},
   cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"'),
   cell('%pip install -q -U "transformers>=4.50,<6" "trl>=0.20" "peft>=0.15" "accelerate>=1.0" "datasets>=3.0" "bitsandbytes>=0.46"'),
   cell(overfit_script + '\n\n' + continuation_script + '\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nadapters = list(Path("/kaggle/input").rglob("overfit_two/adapter/adapter_config.json"))\nif len(files) != 1 or len(adapters) != 1:\n    raise FileNotFoundError(f"Expected one instructed dataset and one 60-step adapter output; found {len(files)} datasets and {len(adapters)} adapters")\nmain(["--dataset", str(files[0]), "--adapter", str(adapters[0].parent), "--output-dir", "/kaggle/working/overfit_two_continued"])'),
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
 'nbformat':4,'nbformat_minor':5
}
(root / 'kaggle' / 'continue_overfit_two.ipynb').write_text(json.dumps(continuation_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

epoch_script = notebook_source('track_validation_epochs')
epoch_script = epoch_script[:epoch_script.rfind('\nif __name__ == "__main__":')]
epoch_book = {
 'cells':[
   {'cell_type':'markdown','metadata':{},'source':['# Held-out rewrite quality across four QLoRA epochs\n', 'Import as a separate **private** Kaggle notebook. Attach only the private instructed 78-pair Dataset, enable `HF_TOKEN`, select a T4 GPU, and allow internet. This trains a fresh Gemma 3 1B QLoRA adapter on the 54 training pairs for four pilot epochs. At baseline and after every epoch, it computes completion-only loss on all 12 validation pairs and deterministically generates rewrites for fixed validation IDs `p008`, `p026`, `p050`, and `p080`. It saves `/kaggle/working/epoch_validation/epoch_validation_report.json`, per-epoch adapters, and `/kaggle/working/epoch_validation/epoch_claim_review.md` with blank claim-preservation checkboxes. The 12 test rows are never used. Surface similarity and copying flags are diagnostics, not semantic claim judgments.\n']},
   cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"'),
   cell('%pip install -q -U "transformers>=4.50,<6" "trl>=0.20" "peft>=0.15" "accelerate>=1.0" "datasets>=3.0" "bitsandbytes>=0.46"'),
   cell(epoch_script + '\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nif len(files) != 1:\n    raise FileNotFoundError(f"Expected one instructed pilot dataset; found {len(files)}")\nmain(["--dataset", str(files[0]), "--output-dir", "/kaggle/working/epoch_validation"])'),
 ],
 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
 'nbformat':4,'nbformat_minor':5
}
(root / 'kaggle' / 'track_validation_epochs.ipynb').write_text(json.dumps(epoch_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

check_script = notebook_source('check_epoch_adapters')
check_script = check_script[:check_script.rfind('\nif __name__ == "__main__":')]
check_import = 'from track_validation_epochs import MODEL_ID, load_pilot, load_hf_token, normalized\n'
assert check_import in check_script
check_script = check_script.replace(check_import, '')
check_book = json.loads(json.dumps(epoch_book))
check_book['cells'][0]['source'] = ['# Saved epoch adapter inference check\n',
 'Attach the instructed pilot Dataset and **version 1 output of `oliverlds/track-validation-epochs`**. Enable `HF_TOKEN`, select T4, and allow internet. This reloads the epoch 2 and epoch 4 adapters for deterministic inference on validation rows `p008` and `p050`, using the same generation settings as the training snapshots. It saves `epoch_adapter_check.json` with earlier and reloaded outputs and match flags. It performs no training.\n']
check_book['cells'][3] = cell(epoch_script + '\n\n' + check_script + '\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nreports = list(Path("/kaggle/input").rglob("epoch_validation_report.json"))\nif len(files) != 1 or len(reports) != 1:\n    raise FileNotFoundError(f"Expected one instructed dataset and one epoch-tracking run output; found {len(files)} datasets and {len(reports)} reports")\nmain(["--dataset", str(files[0]), "--run-dir", str(reports[0].parent)])')
(root / 'kaggle' / 'check_epoch_adapters.ipynb').write_text(json.dumps(check_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

low_lr_book = json.loads(json.dumps(epoch_book))
low_lr_book['cells'][0]['source'] = ['# Four-epoch pilot with a smaller learning rate\n',
 'Run this pilot after the independent adapter check confirms copying/repetition in the saved adapters. Attach **only** the instructed pilot Dataset, enable `HF_TOKEN`, select T4, and allow internet. This starts a fresh adapter with initial learning rate `5e-5` (the previous run used `2e-4`). It keeps the same four epochs, 54/12/12 split, LoRA settings, seed, and fixed validation generations. It saves the report, manual claim worksheet, and epoch adapters under `/kaggle/working/epoch_validation_low_lr`. Test rows are never used.\n']
low_lr_book['cells'][3] = cell(epoch_script + '\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nif len(files) != 1:\n    raise FileNotFoundError(f"Expected one instructed pilot dataset; found {len(files)}")\nmain(["--dataset", str(files[0]), "--learning-rate", "5e-5", "--output-dir", "/kaggle/working/epoch_validation_low_lr"])')
(root / 'kaggle' / 'track_validation_low_lr.ipynb').write_text(json.dumps(low_lr_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

few_shot_script = notebook_source('compare_few_shot')
few_shot_book = {
 'cells': [
   {'cell_type':'markdown','metadata':{},'source':[
     '# Gemma 1B versus 4B: Fama-style rewriting from three examples\n',
     '**Precision correction:** both models now use FP32 compute with 4-bit weights. The original FP16 run produced empty 4B outputs. This version stops immediately on invalid generation scores or empty decoded text and requires 12 nonempty outputs before marking a report complete.\n',
     'Import as a **new private notebook**. Attach only the existing instructed pilot Dataset. Enable `HF_TOKEN`, T4, and internet, then **Save & Run All**. Accept the Hugging Face Gemma license for both `google/gemma-3-1b-it` and `google/gemma-3-4b-it` if needed. The notebook checks access before downloading weights. It runs inference only: three reviewed training demonstrations, identical style instructions, and the same 12 validation passages for both models. Each model runs in a separate process using one T4 and 4-bit weights. No adapters or test examples are used. Outputs: `few_shot_comparison/gemma-3-1b-it.json`, `few_shot_comparison/gemma-3-4b-it.json`, and `few_shot_comparison/claim_style_review.md`. Model loading and generation have not been GPU-tested locally.\n']},
   cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"'),
   cell('%pip install -q -U "transformers>=4.50,<6" "accelerate>=1.0" "bitsandbytes>=0.46" "sentencepiece" "Pillow"'),
   cell('%%writefile /kaggle/working/compare_few_shot.py\n' + few_shot_script),
   cell('from pathlib import Path\nimport subprocess\nimport sys\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nif len(files) != 1:\n    raise FileNotFoundError(f"Expected one instructed Dataset; found {len(files)}")\ncommand = [sys.executable, "/kaggle/working/compare_few_shot.py", "--dataset", str(files[0])]\nsubprocess.run(command + ["--check-access"], check=True)\nfor model_id in ("google/gemma-3-1b-it", "google/gemma-3-4b-it"):\n    subprocess.run(command + ["--model-id", model_id], check=True)\nprint("Completed 24 validation generations; review /kaggle/working/few_shot_comparison/claim_style_review.md")'),
 ],
 'metadata': {'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},
 'nbformat':4,'nbformat_minor':5
}
(root / 'kaggle' / 'compare_few_shot.ipynb').write_text(json.dumps(few_shot_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

large_book = json.loads(json.dumps(few_shot_book))
large_book['cells'][0]['source'] = [
 '# Gemma 3 12B versus the saved 4B baseline\n',
 'Import as a **new private notebook** and select **GPU T4 x2**, internet, and the `HF_TOKEN` Secret. Attach the instructed pilot Dataset and **version 2 output of `oliverlds/compare-few-shot`**. Accept the Hugging Face license for `google/gemma-3-12b-it`. The notebook validates the saved FP32 4B baseline and all prompt hashes before downloading weights. It runs **12B inference only**, with the same instruction, three reviewed demonstrations, and 12 validation inputs. It uses both GPUs with balanced placement, NF4 weights, FP32 compute, and memory headroom. CPU/disk placement is rejected. The first validation generation checks numerical scores, nonempty text, and the output limit before processing the remaining 11. Memory fit still requires confirmation on Kaggle. Outputs are under `/kaggle/working/few_shot_12b`: `12b_hardware_preflight.json`, `gemma-3-12b-it.json`, the copied 4B baseline, and `claim_style_review.md`. No training or test-set evaluation is performed.\n']
large_book['cells'][1] = cell('import os\n# Fresh child processes must see both T4 GPUs.\nos.environ["CUDA_VISIBLE_DEVICES"] = "0,1"')
large_book['cells'][4] = cell('from pathlib import Path\nimport subprocess\nimport sys\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_instructed.jsonl"))\nbaselines = list(Path("/kaggle/input").rglob("gemma-3-4b-it.json"))\nif len(files) != 1 or len(baselines) != 1:\n    raise FileNotFoundError(f"Expected one instructed Dataset and one saved 4B baseline; found {len(files)} datasets and {len(baselines)} baselines")\ncommand = [sys.executable, "/kaggle/working/compare_few_shot.py", "--dataset", str(files[0]), "--model-id", "google/gemma-3-12b-it", "--baseline", str(baselines[0]), "--output-dir", "/kaggle/working/few_shot_12b"]\nsubprocess.run(command + ["--check-access"], check=True)\nsubprocess.run(command, check=True)\nprint("Completed 12B inference; compare with the saved 4B baseline in /kaggle/working/few_shot_12b/claim_style_review.md")')
(root / 'kaggle' / 'compare_few_shot_12b.ipynb').write_text(json.dumps(large_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

smoke4_script = notebook_source('smoke_4b_reviewed')
smoke4_import = 'from compare_few_shot import DEMO_IDS, INSTRUCTION, load_hf_token, make_prompt, normalized, validate_completion\n'
assert smoke4_import in smoke4_script
smoke4_script = smoke4_script.replace(smoke4_import, '')
shared_few_shot = few_shot_script[:few_shot_script.rfind('\nif __name__ == "__main__":')]
smoke4_book = json.loads(json.dumps(few_shot_book))
smoke4_book['cells'][0]['source'] = [
 '# Reviewed-pair Gemma 3 4B QLoRA smoke test\n',
 'Import as a **new private notebook**. Select GPU T4 x2, enable internet and the `HF_TOKEN` Secret, and attach (1) the **new** Dataset file `fama_jensen_1983_pilot_reviewed.jsonl` and (2) **version 2 output of `oliverlds/compare-few-shot`**. This test deliberately uses **one T4 for training**; the second is idle and its memory is not pooled. It trains three unique rows (the longest tokenized eligible training row plus `p031` and `p060`, with duplicate-safe selection) for five optimizer steps, using the same explicit instruction and three frozen demonstrations as the saved 4B baseline. All 48 eligible training rows and 12 validation rows have their token boundaries/labels checked, but only three rows train. The three demonstrations and 12 test rows never train. Targets are completion-masked, base/vision weights are frozen, and gradients and adapter changes are checked. Outputs under `gemma4b_reviewed_smoke/`: `smoke_status.json`, `prepared_label_audit.json`, `training_log.json`, `evaluation.json`, and `adapter/`. Five steps test the pipeline and memory, not style quality. No full pilot starts.\n']
smoke4_book['cells'][2] = cell('%pip install -q -U "transformers>=4.50,<6" "peft>=0.15" "accelerate>=1.0" "datasets>=3.0" "bitsandbytes>=0.46" "sentencepiece" "Pillow"')
smoke4_book['cells'][3] = cell('%%writefile /kaggle/working/smoke_4b_reviewed.py\n' + shared_few_shot + '\n\n' + smoke4_script)
smoke4_book['cells'][4] = cell('from pathlib import Path\nimport subprocess\nimport sys\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_reviewed.jsonl"))\nbaselines = list(Path("/kaggle/input").rglob("gemma-3-4b-it.json"))\nif len(files) != 1 or len(baselines) != 1:\n    raise FileNotFoundError(f"Expected one reviewed Dataset and one saved 4B baseline; found {len(files)} datasets and {len(baselines)} baselines")\ncommand = [sys.executable, "/kaggle/working/smoke_4b_reviewed.py", "--dataset", str(files[0]), "--baseline", str(baselines[0])]\nsubprocess.run(command + ["--check"], check=True)\nsubprocess.run(command, check=True)\nprint("Five-step smoke finished; inspect /kaggle/working/gemma4b_reviewed_smoke/evaluation.json and smoke_status.json")')
(root / 'kaggle' / 'smoke_4b_reviewed.ipynb').write_text(json.dumps(smoke4_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

# Separate helper modules retain normal Python imports and avoid global-name collisions.
pilot4_book = json.loads(json.dumps(smoke4_book))
pilot4_book['cells'][0]['source'] = [
 '# Reviewed Gemma 3 4B: controlled four-epoch pilot\n',
 'Import as a **new private notebook**. Select **GPU T4 x2**, internet, and the enabled **HF_TOKEN** Secret. Attach (1) the reviewed Dataset `fama_jensen_1983_pilot_reviewed.jsonl`, (2) **version 2 output of `oliverlds/compare-few-shot`**, and (3) **version 1 output of `oliverlds/smoke-4b-reviewed`**. This starts a **fresh** rank-8 adapter on all 48 reviewed training pairs, with the frozen explicit instruction and three demonstrations. It uses **one T4**, NF4 weights, FP32 compute, learning rate `5e-5`, batch size 1, accumulation 2, and four epochs (96 optimizer updates). No smoke adapter is loaded. All 12 validation examples are evaluated and generated before training and after every epoch (60 rewrites total); generation can take longer than training. Twelve reserved test rows remain unused. Epoch adapters, incremental `evaluation.json`, `claim_review.md`, label audit, training log, and `pilot_status.json` are saved under `/kaggle/working/reviewed_4b_pilot/`. Claim checkboxes require review; loss and copying flags alone do not establish faithfulness. Run the final cell once per fresh session; it rejects a nonempty output directory.\n']
pilot4_book['cells'][2] = cell('%pip install -q -U "transformers==5.20.1" "peft==0.21.2" "accelerate>=1.0" "datasets>=3.0" "bitsandbytes>=0.46" "sentencepiece" "Pillow"')
pilot4_book['cells'] = pilot4_book['cells'][:3] + [
 cell('%%writefile /kaggle/working/compare_few_shot.py\n' + few_shot_script),
 cell('%%writefile /kaggle/working/smoke_4b_reviewed.py\n' + notebook_source('smoke_4b_reviewed')),
 cell('%%writefile /kaggle/working/track_reviewed_4b.py\n' + notebook_source('track_reviewed_4b')),
 cell('from pathlib import Path\nimport subprocess\nimport sys\n\nfiles = list(Path("/kaggle/input").rglob("fama_jensen_1983_pilot_reviewed.jsonl"))\nbaselines = list(Path("/kaggle/input").rglob("gemma-3-4b-it.json"))\nsmokes = list(Path("/kaggle/input").rglob("gemma4b_reviewed_smoke/smoke_status.json"))\nif len(files) != 1 or len(baselines) != 1 or len(smokes) != 1:\n    raise FileNotFoundError(f"Attach one reviewed Dataset, saved 4B baseline, and successful smoke output; found {len(files)}, {len(baselines)}, {len(smokes)}")\ncommand = [sys.executable, "/kaggle/working/track_reviewed_4b.py", "--dataset", str(files[0]), "--baseline", str(baselines[0]), "--smoke-status", str(smokes[0])]\nsubprocess.run(command + ["--check"], check=True)\nsubprocess.run(command, check=True)\nprint("Review /kaggle/working/reviewed_4b_pilot/evaluation.json and claim_review.md")'),
]
(root / 'kaggle' / 'track_reviewed_4b.ipynb').write_text(json.dumps(pilot4_book, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

corpus_book = {
    'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Multi-paper corpus v1: token audit and base Gemma 3 4B validation\n',
            '**Runtime fixes:** this build resolves the public Hub commit SHA and explicitly '
            'initializes CUDA before resetting memory counters, and checks actual tensor placement '
            'without requiring optional `hf_device_map` metadata. Failure reports retain safe '
            'stack locations and fixed error codes, without remote exception text.\n',
            'Import this as a **new private Kaggle notebook**. Upload the current local '
            '`data/fama_corpus_v1/curated.jsonl` to a private Kaggle Dataset and attach it here; '
            'historical pilot datasets are not accepted. Enable **HF_TOKEN** in Kaggle Secrets, '
            'internet, and **GPU T4 x2**. Accept the Hugging Face license for `google/gemma-3-4b-it`. '
            'Only the first T4 is used. **No TPU, adapter input, or training is required.**\n',
            '\nThis notebook first checks the frozen 313-row dataset hash, paper splits and exact '
            'rewrite instruction. It audits all 313 chat-formatted source/target token lengths and '
            'completion boundaries without truncation, then recommends a sequence limit. The 45 test '
            'examples are tokenized for lengths only: **no test generation or test scoring**. It then '
            'loads base Gemma 3 4B with NF4 weights and FP32 compute and generates exactly the '
            '21 validation inputs, using their stored prompts without demonstrations.\n',
            '\nOutputs under `/kaggle/working/corpus_v1_baseline/`: `token_audit.json`, '
            '`baseline.json`, `baseline_generations.jsonl`, `baseline_manifest.json`, '
            '`claim_style_review.md`, `review_scores.jsonl`, and `run_status.json`. Model-loading '
            'and generation heartbeats print every 30 seconds. Partial outputs are saved after '
            'each example. Empty, invalid-score, or cut-off outputs stop the run with incomplete '
            'status. Claim preservation and style scores remain blank for separate review.\n',
            '\nRun once per fresh session. A nonempty output directory is rejected to protect '
            'the saved baseline. This notebook has passed local CPU checks; actual token lengths, '
            'GPU memory fit and runtime results still require this Kaggle execution.\n']},
        cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
             'os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\n'
             'os.environ["TOKENIZERS_PARALLELISM"] = "false"'),
        cell('%pip install -q -U "transformers==5.20.1" "accelerate>=1.0" '
             '"bitsandbytes>=0.46" "sentencepiece" "Pillow"'),
        cell('%%writefile /kaggle/working/corpus_baseline.py\n' + notebook_source('corpus_baseline')),
        cell('from pathlib import Path\nimport subprocess\nimport sys\n\n'
             'files = list(Path("/kaggle/input").rglob("curated.jsonl"))\n'
             'if len(files) != 1:\n'
             '    raise FileNotFoundError(f"Attach exactly one corpus-v1 curated.jsonl; found {len(files)}")\n'
             'command = [sys.executable, "/kaggle/working/corpus_baseline.py", "--dataset", str(files[0])]\n'
             'subprocess.run(command + ["--check"], check=True)\n'
             'subprocess.run(command, check=True)'),
        cell('import json\nfrom pathlib import Path\n\n'
             'output = Path("/kaggle/working/corpus_v1_baseline")\n'
             'status = json.loads((output / "run_status.json").read_text())\n'
             'baseline = json.loads((output / "baseline.json").read_text())\n'
             'audit = json.loads((output / "token_audit.json").read_text())\n'
             'if not status["complete"] or not baseline["complete"] or len(baseline["results"]) != 21:\n'
             '    raise RuntimeError("Baseline is incomplete; inspect run_status.json and partial outputs")\n'
             'print("Completed 21 validation generations; no training or test generation/scoring.")\n'
             'print("Rows length-audited:", audit["rows_audited"])\n'
             'print("Recommended max_length:", audit["recommended_max_length"])\n'
             'print("Review claims and style separately in claim_style_review.md; download all output files.")'),
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python'}},
    'nbformat': 4, 'nbformat_minor': 5,
}
(root / 'kaggle' / 'corpus_v1_baseline.ipynb').write_text(
    json.dumps(corpus_book, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

corpus_smoke_book = {
    'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Corpus v1: three-example, five-update Gemma 3 4B QLoRA smoke test\n',
            'Import as a **new private notebook**. Attach the same private Dataset containing '
            '`curated.jsonl` and, through **Add Input → Notebook Output Files**, the completed '
            '**oliverlds/corpus_v1_baseline version 4** output. Enable **HF_TOKEN** in Kaggle Secrets '
            'and internet; select **GPU T4 x2**. Only the first T4 is exposed. No TPU is needed.\n',
            '\nThis is a pipeline/memory test: **three training examples, exactly five optimizer '
            'updates**, batch size 1 with two microbatches per update, sequence limit 1024 without '
            'truncation, NF4 weights/FP32 compute, language q/v LoRA rank 8. Frozen baseline hashes '
            'and the original Gemma revision are checked before any model download. The three '
            'longest training pairs are used; the two longest validation pairs are used only '
            'for teacher-forced loss before/after training and after adapter reload. No test rows '
            'are tokenized, trained, generated or scored by this notebook.\n',
            '\nPrompt labels are masked; the actual training batch must preserve every target '
            'token and match the verified baseline audit. Losses/gradients must remain finite; '
            'adapter weights must change. The saved adapter is unloaded without merging and '
            'reloaded from disk onto the same frozen 4B base. Tensor hashes and held-out losses '
            'must match before completion is accepted.\n',
            '\nOutputs: `/kaggle/working/corpus_v1_smoke/`, including `smoke_status.json`, '
            '`prepared_label_audit.json`, `training_log.json`, `adapter/`, '
            '`adapter_tensor_hashes.json`, and `evaluation.json`. The latter compares original, '
            'fine-tuned and reloaded losses on just two validation rows. A loss decrease is not '
            'required, and no style improvement is claimed. No generations or full training '
            'are included. Use a fresh session; existing results are never overwritten.\n']},
        cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
             'os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\n'
             'os.environ["TOKENIZERS_PARALLELISM"] = "false"'),
        cell('%pip install -q -U "transformers==5.20.1" "peft==0.21.2" '
             '"accelerate>=1.0" "bitsandbytes>=0.46" "sentencepiece" "Pillow"'),
        cell('%%writefile /kaggle/working/corpus_baseline.py\n' + notebook_source('corpus_baseline')),
        cell('%%writefile /kaggle/working/corpus_smoke.py\n' + notebook_source('corpus_smoke')),
        cell('from pathlib import Path\nimport subprocess\nimport sys\n\n'
             'datasets = list(Path("/kaggle/input").rglob("curated.jsonl"))\n'
             'manifests = list(Path("/kaggle/input").rglob("baseline_manifest.json"))\n'
             'if len(datasets) != 1 or len(manifests) != 1:\n'
             '    raise FileNotFoundError("Attach exactly one curated.jsonl and baseline version 4 output")\n'
             'command = [sys.executable, "/kaggle/working/corpus_smoke.py", "--dataset", str(datasets[0]),\n'
             '           "--baseline-dir", str(manifests[0].parent)]\n'
             'subprocess.run(command + ["--check"], check=True)\n'
             'subprocess.run(command, check=True)'),
        cell('import json\nimport sys\nfrom pathlib import Path\n'
             'sys.path.insert(0, "/kaggle/working")\n'
             'from corpus_smoke import validate_completion\n'
             'output = Path("/kaggle/working/corpus_v1_smoke")\n'
             'status = json.loads((output / "smoke_status.json").read_text())\n'
             'evaluation = json.loads((output / "evaluation.json").read_text())\n'
             'validate_completion(status, evaluation)\n'
             'print("Verified: 3 training examples, 5 optimizer updates, adapter save/reload.")\n'
             'for key in ("original_validation_loss", "fine_tuned_validation_loss", "reloaded_validation_loss"):\n'
             '    print(key, evaluation[key])\n'
             'print("Two-row loss check only; no learning-quality conclusion or test scoring.")\n'
             'print("Download the complete corpus_v1_smoke output directory.")'),
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python'}},
    'nbformat': 4, 'nbformat_minor': 5,
}
(root / 'kaggle' / 'corpus_v1_smoke.ipynb').write_text(
    json.dumps(corpus_smoke_book, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

corpus_pilot_book = {
    'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Corpus v1: one-epoch Gemma 3 4B QLoRA pilot\n',
            'Import as a **new private notebook**. Attach the private Dataset with `curated.jsonl`, '
            '**corpus_v1_baseline version 4 output**, and **corpus_v1_smoke version 1 output**. '
            'Use **Add Input → Notebook Output Files** for the two notebook outputs. Enable '
            '**HF_TOKEN** in Secrets and internet; select **GPU T4 x2**. Only the first T4 is used.\n',
            '\nThis bounded pilot trains a **fresh rank-8 language q/v LoRA adapter** on base Gemma '
            '3 4B for **exactly one epoch over 247 training pairs**. It does not resume the smoke '
            'adapter. Corpus, baseline and successful smoke hashes are checked before loading weights. '
            'NF4/FP32, learning rate 5e-5, microbatch 1, accumulation 2, seed 42, limit 1024 without '
            'truncation, no packing. Each pair is seen once in shuffled order: **124 updates**, '
            'with the last update correctly normalized for its single microbatch. No automatic '
            'second epoch or extended training is included.\n',
            '\nCompletion-only labels are verified for all 247 training and 21 validation pairs. '
            'All 21 validation losses are measured before/after training and after adapter reload. '
            'The epoch adapter is saved and reloaded before greedy validation generation, using '
            'the exact stored prompts and 512-token budget from the frozen base comparator. '
            'No test pair is tokenized, trained, generated or scored here.\n',
            '\nOutputs under `/kaggle/working/corpus_v1_pilot/`: `pilot_status.json`, '
            '`evaluation.json`, `comparison.json`, `validation_generations.jsonl`, '
            '`adapter/`, `pilot_manifest.json`, label audit and training logs. '
            '`claim_style_review.md` varies A/B order, `comparison_key.json` contains the mapping, '
            'and `review_scores.jsonl` has blank separate claim/style fields. Masking labels is '
            'not independent blinded evaluation. Copy/length statistics are descriptive only.\n',
            '\nUpdates and generations are saved incrementally; a failed run stays incomplete. '
            'Heartbeats print every 30 seconds during slow operations. Use a fresh session. '
            'Based on the smoke timings, budget roughly **25–40 minutes**; actual runtime varies '
            'with pair lengths, downloads and generation lengths. A lower loss is not proof '
            'of improved claims or style.\n']},
        cell('import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
             'os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"\n'
             'os.environ["TOKENIZERS_PARALLELISM"] = "false"'),
        cell('%pip install -q -U "transformers==5.20.1" "peft==0.21.2" '
             '"accelerate==1.15.0" "bitsandbytes==0.50.2" "huggingface-hub==1.33.0" '
             '"sentencepiece" "Pillow"'),
        cell('%%writefile /kaggle/working/corpus_baseline.py\n' + notebook_source('corpus_baseline')),
        cell('%%writefile /kaggle/working/corpus_smoke.py\n' + notebook_source('corpus_smoke')),
        cell('%%writefile /kaggle/working/corpus_pilot.py\n' + notebook_source('corpus_pilot')),
        cell('from pathlib import Path\nimport subprocess\nimport sys\n\n'
             'root = Path("/kaggle/input")\n'
             'datasets = list(root.rglob("curated.jsonl"))\n'
             'baselines = list(root.rglob("baseline_manifest.json"))\n'
             'smokes = list(root.rglob("smoke_status.json"))\n'
             'if len(datasets) != 1 or len(baselines) != 1 or len(smokes) != 1:\n'
             '    raise FileNotFoundError("Attach exactly one corpus Dataset, baseline v4 output, and smoke v1 output")\n'
             'command = [sys.executable, "/kaggle/working/corpus_pilot.py", "--dataset", str(datasets[0]),\n'
             '           "--baseline-dir", str(baselines[0].parent), "--smoke-dir", str(smokes[0].parent)]\n'
             'subprocess.run(command + ["--check"], check=True)\n'
             'subprocess.run(command, check=True)'),
        cell('import json\nimport sys\nfrom pathlib import Path\n'
             'sys.path.insert(0, "/kaggle/working")\n'
             'from corpus_pilot import load_inputs, validate_completion\n'
             'groups, audit, original = load_inputs(datasets[0], baselines[0].parent, smokes[0].parent)\n'
             'output = Path("/kaggle/working/corpus_v1_pilot")\n'
             'status = json.loads((output / "pilot_status.json").read_text())\n'
             'evaluation = json.loads((output / "evaluation.json").read_text())\n'
             'comparison = json.loads((output / "comparison.json").read_text())\n'
             'validate_completion(status, evaluation, comparison, groups, original)\n'
             'print("Verified: one epoch, 247 examples, 124 updates, adapter reload, 21 validation generations.")\n'
             'print("Original validation loss:", evaluation["original_validation_loss"])\n'
             'print("Adapted validation loss:", evaluation["fine_tuned_validation_loss"])\n'
             'print("Review claim preservation and style separately; keep comparison_key.json closed while scoring.")\n'
             'print("Download the complete corpus_v1_pilot output directory.")'),
    ],
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python'}},
    'nbformat': 4, 'nbformat_minor': 5,
}
(root / 'kaggle' / 'corpus_v1_pilot.ipynb').write_text(
    json.dumps(corpus_pilot_book, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
