"""Token-length preflight and inference-only 4B baseline for corpus v1."""

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
import re
import threading
import time
from pathlib import Path

from fama_style.prompting.corpus import FORMAT, INSTRUCTION, make_prompt

MODEL_ID = "google/gemma-3-4b-it"
DATASET_SHA256 = "c24e2d708884877ce61ea212c4d6e442261728be499ff2d15ee7826887a4299f"
COUNTS = {"train": 247, "validation": 21, "test": 45}
PAPERS = {"F1965": ("train", 30), "F1980": ("train", 6),
          "FF2000": ("train", 40), "FJ1983": ("train", 50),
          "ME1997": ("train", 60), "NL2003": ("train", 40), "VL2018": ("train", 21),
          "FF1997": ("validation", 21), "FF2014": ("test", 45)}
CONTEXT_BUDGET = 4096
IMPLEMENTATION = "corpus_baseline_v4_actual_device_check"


def validate_model_placement(model):
    """Check actual tensors; hf_device_map is optional loader metadata."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None and (not isinstance(device_map, dict) or not device_map
            or any(str(device) not in ("0", "cuda:0") for device in device_map.values())):
        raise PreflightError("model_device_map_not_first_gpu")
    counts = {"parameters": 0, "buffers": 0}
    for kind, tensors in (("parameters", model.parameters()), ("buffers", model.buffers())):
        for tensor in tensors:
            if tensor.device.type != "cuda" or tensor.device.index != 0:
                raise PreflightError("model_tensor_not_first_gpu")
            counts[kind] += 1
    if not counts["parameters"]:
        raise PreflightError("model_has_no_parameters")
    embedding_device = model.get_input_embeddings().weight.device
    if embedding_device.type != "cuda" or embedding_device.index != 0:
        raise PreflightError("model_embedding_not_first_gpu")
    return {"device": "cuda:0", "device_map_present": device_map is not None,
            "parameter_tensors_checked": counts["parameters"], "buffer_tensors_checked": counts["buffers"]}


def initialize_cuda(torch):
    """Initialize the allocator before resetting its counters in a fresh process."""
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats(0)


def safe_failure_details(exc):
    """Retain stack locations, never exception messages, source lines or locals.

    Remote errors may include authenticated URLs. Only a closed set of labels,
    numeric line numbers and local/package stack locations may leave this helper.
    """
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename.replace("\\", "/")
        category = None
        if filename == __file__:
            category = "corpus_baseline"
        else:
            for package in ("torch", "transformers", "bitsandbytes", "accelerate", "huggingface_hub"):
                if "/" + package + "/" in filename:
                    category = package
                    break
        if category:
            frames.append({"component": category, "line": tb.tb_lineno})
        tb = tb.tb_next
    # Compare internally; never serialize even a redacted remote message.
    text = str(exc)
    code = "unclassified_runtime_failure"
    for marker, label in (
        ("invalid argument to reset_peak_memory_stats", "cuda_allocator_not_initialized"),
        ("CUDA out of memory", "cuda_out_of_memory"),
        ("no kernel image is available", "cuda_kernel_not_supported"),
        ("CUDA driver version is insufficient", "cuda_driver_incompatible"),
    ):
        if marker in text:
            code = label
            break
    if "Invalid device argument" in text and "did you call init?" in text:
        code = "cuda_allocator_not_initialized"
    return {"error_code": code, "stack_locations": frames}


class PreflightError(ValueError):
    """A known local failure code, with no remote exception text or credentials."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def load_pinned_config(api, config_class, token, progress=None):
    """Resolve a public Hub SHA, then load exactly that configuration revision.

    Transformers 5.20 discards the old private config._commit_hash attribute.
    Config, processor and weights must instead receive the same public revision.
    """
    info = api.model_info(MODEL_ID, revision="main", timeout=30)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise PreflightError("hub_revision_missing_or_invalid")
    if progress:
        progress("loading_model_config", model_id=MODEL_ID, model_revision=revision)
    config = config_class.from_pretrained(MODEL_ID, revision=revision, token=token)
    return config, revision


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_rows(rows):
    if len(rows) != 313 or len({r["id"] for r in rows}) != 313:
        raise ValueError("Expected 313 unique corpus-v1 rows")
    if dict(collections.Counter(r["split"] for r in rows)) != COUNTS:
        raise ValueError("Corpus split counts changed")
    if dict(collections.Counter(r["paper_key"] for r in rows)) != {p: n for p, (_, n) in PAPERS.items()}:
        raise ValueError("Corpus paper counts changed")
    for row in rows:
        if (row["split"] != PAPERS[row["paper_key"]][0] or row["prompt_format"] != FORMAT
                or row["prompt"] != make_prompt(row["plain_english"])
                or row["completion"] != row["original_paragraph"]
                or not row["plain_english"].strip() or not row["completion"].strip()):
            raise ValueError("Changed prompt format, paper split, or source/target alignment")
    return {split: [r for r in rows if r["split"] == split] for split in COUNTS}


def load_corpus(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATASET_SHA256:
        raise ValueError("Attach the exact current data/fama_corpus_v1/curated.jsonl; SHA-256 mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, validate_rows(rows)


def load_hf_token():
    if Path("/kaggle").exists():
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            raise RuntimeError("Enable the Kaggle Secret named HF_TOKEN for this notebook") from None
    else:
        token = os.environ.get("HF_TOKEN")
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError("HF_TOKEN is unavailable; configure it outside notebook code and the repository")
    return token


def user_message(row):
    return {"role": "user", "content": [{"type": "text", "text": row["prompt"]}]}


def input_ids(batch):
    value = batch["input_ids"]
    value = value.tolist() if hasattr(value, "tolist") else value
    return value[0] if value and isinstance(value[0], list) else value


def prepare_pair(processor, row):
    """Verify an exact completion boundary, without truncation or exposing token text."""
    user = user_message(row)
    assistant = {"role": "assistant", "content": [{"type": "text", "text": row["completion"]}]}
    prefix = processor.apply_chat_template([user], tokenize=True, add_generation_prompt=True,
                                          truncation=False, return_dict=True, return_tensors="pt")
    full = processor.apply_chat_template([user, assistant], tokenize=True, add_generation_prompt=False,
                                        truncation=False, return_dict=True, return_tensors="pt")
    prefix_ids, ids = input_ids(prefix), input_ids(full)
    if not prefix_ids or ids[:len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids):
        raise ValueError("Chat-template prompt/target boundary mismatch; no truncation permitted")
    target_ids = ids[len(prefix_ids):]
    decoded = processor.tokenizer.decode(target_ids, skip_special_tokens=True)
    if " ".join(decoded.split()) != " ".join(row["completion"].split()):
        raise ValueError("Decoded completion tokens do not match the complete target")
    audit = {"id": row["id"], "paper_key": row["paper_key"], "split": row["split"],
             "prompt_tokens": len(prefix_ids), "target_tokens": len(target_ids), "full_tokens": len(ids),
             "prompt_sha256": digest(row["prompt"]),
             "prefix_token_ids_sha256": digest(json.dumps(prefix_ids)),
             "target_text_sha256": digest(row["completion"]),
             "boundary_matches": True, "decoded_target_matches": True, "truncated": False}
    return audit


def summarize_audit(records):
    if len(records) != 313 or dict(collections.Counter(r["split"] for r in records)) != COUNTS:
        raise ValueError("Token audit must cover every corpus row")
    maximum = max(r["full_tokens"] for r in records)
    recommended = next((n for n in (1024, 1536, 2048, 3072, 4096) if n >= maximum), None)
    validation = [r for r in records if r["split"] == "validation"]
    generation_budget = max(512, ((max(r["target_tokens"] for r in validation) + 64 + 127) // 128) * 128)
    return {"model_id": MODEL_ID, "dataset_sha256": DATASET_SHA256, "rows_audited": 313,
            "split_counts": COUNTS, "test_rows_length_audited": 45,
            "test_rows_generated": 0, "test_rows_scored": 0,
            "longest_full_sequence": maximum, "recommended_max_length": recommended,
            "recommendation_scope": "Fits all 313 labeled sequences; this is not a GPU memory-fit result",
            "baseline_max_new_tokens": generation_budget,
            "fits_context_budget": recommended is not None and generation_budget <= 1024 and
                all(r["prompt_tokens"] + generation_budget <= CONTEXT_BUDGET for r in validation),
            "limits": {str(n): {s: sum(r["full_tokens"] > n for r in records if r["split"] == s)
                                for s in COUNTS} for n in (1024, 1536, 2048, 3072, 4096)},
            "longest_train_ids": [r["id"] for r in sorted(
                (r for r in records if r["split"] == "train"), key=lambda r: (-r["full_tokens"], r["id"]))[:3]],
            "records": records}


def validate_result(result):
    if not result["output"].strip() or result["hit_output_limit"] or result["output_tokens"] <= 0:
        raise ValueError("Empty or cut-off generation; baseline remains incomplete")


def validate_baseline(report, validation, require_complete=True):
    if (report["model_id"] != MODEL_ID or report["dataset_sha256"] != DATASET_SHA256
            or report["prompt_format"] != FORMAT or report["instruction"] != INSTRUCTION
            or report["training_called"] is not False or report["test_rows_generated"] != 0
            or report["test_rows_scored"] != 0 or len(report["results"]) != 21
            or (require_complete and report.get("complete") is not True)
            or report.get("quantization") != "NF4 double quantization, FP32 compute"
            or report.get("demonstrations") != []
            or report.get("generation", {}).get("do_sample") is not False
            or not 0 < report.get("generation", {}).get("max_new_tokens", 0) <= 1024
            or not report.get("model_revision")
            or [r["id"] for r in report["results"]] != [r["id"] for r in validation]):
        raise ValueError("Baseline must contain exactly the frozen validation rows, in order")
    for result, row in zip(report["results"], validation):
        validate_result(result)
        if (result["prompt_sha256"] != digest(row["prompt"]) or result["source"] != row["plain_english"]
                or result["reference"] != row["completion"]):
            raise ValueError("Baseline output provenance does not match validation input")


def save_outputs(output_dir, report, validation):
    write_json(output_dir / "baseline.json", report)
    (output_dir / "baseline_generations.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in report["results"]), encoding="utf-8")
    results = {r["id"]: r for r in report["results"]}
    lines = ["# Corpus-v1 base-model validation review", "",
             "Build the claim inventory from the plain-English source first. Sentence splits below are",
             "reading aids, not atomic claim labels. Judge preservation separately from style.",
             "References are for review only; the model received only the stored source prompt.", ""]
    forms = []
    for row in validation:
        generated = results.get(row["id"], {})
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", row["plain_english"])
        lines.extend([f"## {row['id']}", "", "### Plain-English source", "", row["plain_english"], "",
                      "### Source reading aids", ""])
        lines.extend(f"- {s}" for s in sentences)
        lines.extend(["", "- Claim inventory: ", "", "### Base output", "",
                      generated.get("output", "Pending generation"), "", "### Original reference", "",
                      row["completion"], "", "### Claim preservation", "",
                      "- [ ] All substantive claims retained", "- [ ] No unsupported claims added",
                      "- [ ] Actors, causality, numbers, conditions and qualifiers preserved",
                      "- Added / lost / changed claims: ", "- Claim preservation score (0–4): ", "",
                      "### Style (score separately; 0–2 each)", "",
                      "- Precise economic terminology: ", "- Explicit analytical/conditional structure: ",
                      "- Concise exposition resembling the reference: ", "- Exact input copying: ",
                      "- Style evidence and limitations: ", ""])
        forms.append({"id": row["id"], "claim_inventory": [], "added_claims": [], "lost_claims": [],
                      "changed_claims": [], "claim_preservation_score": None,
                      "style": {"economic_terminology": None, "analytical_structure": None, "concision": None},
                      "review_status": "pending_manual_review"})
    (output_dir / "claim_style_review.md").write_text("\n".join(lines), encoding="utf-8")
    # Created once, so incremental saves never overwrite a review entered during a run.
    form_path = output_dir / "review_scores.jsonl"
    if not form_path.exists():
        form_path.write_text("".join(json.dumps(r) + "\n" for r in forms), encoding="utf-8")


def run_training(rows, groups, output_dir, audit_only=False):
    """Inference only. The name follows the project's GPU-import entry-point convention."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is nonempty; use a fresh Kaggle session or a new output path")
    output_dir.mkdir(parents=True, exist_ok=True)
    status = {"complete": False, "phase": "runtime_start", "training_called": False,
              "test_rows_generated": 0, "test_rows_scored": 0, "dataset_sha256": DATASET_SHA256,
              "implementation": IMPLEMENTATION}
    report = None
    records = []

    def phase(name, **extra):
        status.update(phase=name, **extra)
        write_json(output_dir / "run_status.json", status)
        print(f"Phase: {name}", flush=True)

    def load_with_heartbeat(loader, description):
        stop = threading.Event()
        started = time.monotonic()
        def heartbeat():
            while not stop.wait(30):
                print(f"{description}: still working ({int(time.monotonic() - started)} seconds)", flush=True)
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            return loader()
        finally:
            stop.set()
            thread.join(timeout=1)

    try:
        phase("loading_runtime")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
        import torch
        from transformers import (AutoConfig, AutoProcessor, BitsAndBytesConfig,
                                  Gemma3ForConditionalGeneration, LogitsProcessor, LogitsProcessorList)
        from huggingface_hub import HfApi
        if not audit_only and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError("Select Kaggle GPU T4 x2; this notebook exposes and uses only its first GPU")
        phase("checking_secret")
        token = load_hf_token()
        phase("resolving_model_revision")
        config, revision = load_pinned_config(HfApi(token=token), AutoConfig, token, progress=phase)
        phase("loading_processor", model_id=MODEL_ID, model_revision=revision)
        processor = load_with_heartbeat(lambda: AutoProcessor.from_pretrained(
            MODEL_ID, revision=revision, token=token), "Processor download/load")
        tokenizer = processor.tokenizer
        phase("auditing_all_token_lengths")
        for index, row in enumerate(rows, 1):
            status["current_id"] = row["id"]
            records.append(prepare_pair(processor, row))
            if index % 50 == 0 or index == len(rows):
                print(f"Audited token lengths: {index}/{len(rows)}", flush=True)
        audit = summarize_audit(records)
        template = getattr(processor, "chat_template", None) or tokenizer.chat_template
        audit.update(model_revision=revision, chat_template_sha256=digest(json.dumps(template, sort_keys=True)),
                     tokenizer_class=type(tokenizer).__name__,
                     versions={name: importlib.metadata.version(name) for name in
                               ("torch", "transformers", "tokenizers", "accelerate", "huggingface-hub")},
                     implementation=IMPLEMENTATION)
        write_json(output_dir / "token_audit.json", audit)
        if not audit["fits_context_budget"]:
            raise ValueError("Sequence or generation budget exceeds preflight limits; token audit saved")
        print(f"No truncation: recommended max_length={audit['recommended_max_length']}; "
              f"longest sequence={audit['longest_full_sequence']}", flush=True)
        if audit_only:
            status["complete"] = True
            phase("token_audit_complete", model_loaded=False)
            return
        phase("initializing_cuda")
        initialize_cuda(torch)
        phase("loading_4b_model")
        model = load_with_heartbeat(lambda: Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=revision, token=token, config=config,
            dtype=torch.float32, device_map={"": 0}, attn_implementation="sdpa",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)), "4B model download/load")
        phase("checking_model_placement")
        placement = validate_model_placement(model)
        write_json(output_dir / "model_placement.json", placement)
        model.eval()
        model.requires_grad_(False)
        budget = audit["baseline_max_new_tokens"]
        by_id = {r["id"]: r for r in records}
        report = {"complete": False, "model_id": MODEL_ID, "model_revision": revision,
                  "implementation": IMPLEMENTATION,
                  "dataset_sha256": DATASET_SHA256, "prompt_format": FORMAT, "instruction": INSTRUCTION,
                  "training_called": False, "test_rows_length_audited": 45, "test_rows_generated": 0,
                  "test_rows_scored": 0, "validation_rows": 21, "versions": audit["versions"],
                  "chat_template_sha256": audit["chat_template_sha256"],
                  "quantization": "NF4 double quantization, FP32 compute", "seed": 42,
                  "demonstrations": [], "finite_generation_scores_checked": True,
                  "model_placement": placement,
                  "generation": {"do_sample": False, "max_new_tokens": budget, "use_cache": True},
                  "recommended_max_length": audit["recommended_max_length"], "results": []}
        save_outputs(output_dir, report, groups["validation"])

        class FiniteScores(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if bool(torch.isnan(scores).any() | torch.isposinf(scores).any()
                        | ~torch.isfinite(scores).any(dim=-1).all()):
                    raise RuntimeError("Invalid generation scores; baseline remains incomplete")
                return scores

        for index, row in enumerate(groups["validation"], 1):
            phase("validation_generation", saved_generations=index - 1, current_id=row["id"])
            inputs = processor.apply_chat_template([user_message(row)], tokenize=True, add_generation_prompt=True,
                truncation=False, return_dict=True, return_tensors="pt")
            ids = input_ids(inputs)
            if digest(json.dumps(ids)) != by_id[row["id"]]["prefix_token_ids_sha256"]:
                raise ValueError("Generation prompt differs from audited prompt")
            if len(ids) + budget > CONTEXT_BUDGET:
                raise ValueError("Generation context over budget; no truncation permitted")
            inputs = {k: v.to(model.get_input_embeddings().weight.device) for k, v in inputs.items()}
            started = time.monotonic()
            with torch.inference_mode():
                tokens = load_with_heartbeat(lambda: model.generate(**inputs, do_sample=False,
                    max_new_tokens=budget, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                    remove_invalid_values=False, logits_processor=LogitsProcessorList([FiniteScores()])),
                    f"Validation generation {index}/21")
            generated = tokens[0, len(ids):]
            output = tokenizer.decode(generated, skip_special_tokens=True).strip()
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            result = {"id": row["id"], "paper_key": row["paper_key"], "split": "validation",
                      "source": row["plain_english"], "reference": row["completion"], "output": output,
                      "prompt_sha256": digest(row["prompt"]), "input_tokens": len(ids),
                      "output_tokens": int(generated.numel()), "first_generated_token_ids": generated[:16].tolist(),
                      "hit_output_limit": generated.numel() >= budget and
                          (not generated.numel() or int(generated[-1]) not in eos),
                      "copied_source": " ".join(output.split()) == " ".join(row["plain_english"].split()),
                      "generation_seconds": time.monotonic() - started, "claim_review_status": "pending_manual_review"}
            report["results"].append(result)
            save_outputs(output_dir, report, groups["validation"])
            validate_result(result)
            print(f"Saved valid validation output {index}/21 ({result['output_tokens']} tokens)", flush=True)
        validate_baseline(report, groups["validation"], require_complete=False)
        report["memory"] = {"gpu": torch.cuda.get_device_name(0),
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
        report["complete"] = True
        save_outputs(output_dir, report, groups["validation"])
        artifacts = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
                     for name in ("baseline.json", "baseline_generations.jsonl", "token_audit.json")}
        write_json(output_dir / "baseline_manifest.json", {"model_id": MODEL_ID, "model_revision": revision,
                   "dataset_sha256": DATASET_SHA256, "files": artifacts})
        status["complete"] = True
        phase("baseline_complete", saved_generations=21)
    except Exception as exc:
        failed_phase = status["phase"]
        status.update(complete=False, phase="failed", failed_phase=failed_phase, error_class=type(exc).__name__)
        status.update(safe_failure_details(exc))
        if isinstance(exc, PreflightError):
            status["error_code"] = exc.code
        if failed_phase == "auditing_all_token_lengths":
            write_json(output_dir / "token_audit_partial.json", {"complete": False,
                       "rows_audited": len(records), "failed_id": status.get("current_id"),
                       "dataset_sha256": DATASET_SHA256, "records": records})
        hints = {
            "checking_secret": "Enable the HF_TOKEN Kaggle Secret for this notebook.",
            "resolving_model_revision": "Check internet and Hugging Face model access; this is a Hub revision lookup.",
            "loading_model_config": "Confirm the Gemma license is accepted, model access is permitted, and internet is enabled.",
            "initializing_cuda": "CUDA initialization or memory-counter setup failed before weight loading; inspect the safe stack locations.",
            "loading_4b_model": "Weight loading failed; inspect the safe stack locations and error code. No raw remote exception text is saved.",
        }
        if failed_phase in hints:
            status["hint"] = hints[failed_phase]
        write_json(output_dir / "run_status.json", status)
        # Remote exception messages/URLs can contain credentials. Retain phase/class only.
        hint = " " + status["hint"] if "hint" in status else ""
        raise RuntimeError(f"Preflight stopped in {failed_phase} ({type(exc).__name__}); "
                           "see run_status.json and any saved partial artifacts. No training was called."
                           + hint) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/corpus_v1_baseline")
    parser.add_argument("--check", action="store_true", help="CPU-only schema/hash/split checks; no ML imports")
    parser.add_argument("--audit-only", action="store_true", help="Download processor/tokenizer only; skip model weights")
    args = parser.parse_args(argv)
    rows, groups = load_corpus(args.dataset)
    print("Validated frozen corpus: 247 train / 21 validation / 45 test")
    if args.check:
        print("Dry-run OK. No credentials accessed, model downloaded, generation or training performed.")
        return
    run_training(rows, groups, Path(args.output_dir), args.audit_only)


if __name__ == "__main__":
    main()
