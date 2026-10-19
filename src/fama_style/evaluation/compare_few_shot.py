"""Inference-only comparison of Gemma 3 1B, 4B, and 12B with reviewed examples."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from fama_style.prompting.few_shot import INSTRUCTION, make_prompt

MODEL_IDS = ("google/gemma-3-1b-it", "google/gemma-3-4b-it", "google/gemma-3-12b-it")
DEMO_IDS = ("p014", "p018", "p075")
SOURCE_EDITS = {
    "p018": (("writing and enforcing agreements costs money",
              "writing and enforcing agreements is costly"),),
    "p075": (("The main aim is to explain why organizations survive when financial ownership and decision-making authority are separated.",
              "The main aim is to explain the factors that help organizations survive when financial ownership and decision-making authority are separated, a distinction often described imprecisely as ownership and control."),),
}


def normalized(text):
    return " ".join(text.split())


def validate_completion(report):
    if len(report["results"]) != 12 or any(not r["output"].strip() for r in report["results"]):
        raise RuntimeError("Comparison requires 12 nonempty generations; report remains incomplete")


def validate_baseline(path, dataset_path, demos, validation):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_completion(report)
    if (report.get("model_id") != MODEL_IDS[1] or report.get("complete") is not True
            or report.get("training_called") is not False or report.get("test_rows_used") != 0
            or report.get("quantization") != "NF4 double quantization, FP32 compute"
            or report.get("dataset_sha256") != hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
            or report.get("instruction") != INSTRUCTION or report.get("demonstrations") != demos
            or report.get("generation") != {"do_sample": False, "max_new_tokens": 512}):
        raise ValueError("Attach the completed FP32 4B report from compare-few-shot version 2")
    for old, row in zip(report["results"], validation):
        prompt = make_prompt(demos, row["plain_english"])
        if (old.get("source_ids") != row["source_ids"]
                or old.get("source") != row["plain_english"]
                or old.get("reference") != row["completion"]
                or old.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest()
                or old.get("hit_output_limit") is not False):
            raise ValueError("4B baseline prompts, sources, or generation validity do not match")
    return report


def validate_gpu_placement(device_map):
    if not device_map:
        raise RuntimeError("Missing model device map")
    devices = {str(value) for value in device_map.values()}
    if not devices <= {"0", "1", "cuda:0", "cuda:1"}:
        raise RuntimeError("12B must fit on both GPUs without CPU or disk offload")
    if {value.split(":")[-1] for value in devices} != {"0", "1"}:
        raise RuntimeError("Expected model layers on both GPUs; inspect device map")


def validate_probe(result):
    if not result["output"].strip() or result["hit_output_limit"]:
        raise RuntimeError("First-generation check failed: empty or output-limited text")


def load_experiment(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    groups = {split: [r for r in rows if r.get("split") == split]
              for split in ("train", "validation", "test")}
    if len(rows) != 78 or [len(groups[k]) for k in groups] != [54, 12, 12]:
        raise ValueError("Expected the instructed 78-pair Dataset with 54/12/12 splits")
    for row in rows:
        if row.get("prompt_format") != "rewrite_v1" or any(
                not isinstance(row.get(k), str) or not row[k].strip()
                for k in ("plain_english", "completion")):
            raise ValueError("Expected nonempty instructed source/target rows")
    demos = []
    for key in DEMO_IDS:
        matches = [row for row in groups["train"] if row.get("source_ids") == [key]]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one training demonstration {key}")
        row = matches[0]
        source = row["plain_english"]
        for old, new in SOURCE_EDITS.get(key, ()):
            if source.count(old) != 1:
                raise ValueError(f"Reviewed source text changed for {key}; review again")
            source = source.replace(old, new)
        demos.append({"source_id": key, "source": source,
                      "original_source": row["plain_english"],
                      "target": normalized(row["completion"])})
    sources = [normalized(row["plain_english"]) for row in groups["validation"]]
    if len(set(sources)) != 12 or any(normalized(d["source"]) in sources or
            normalized(d["original_source"]) in sources for d in demos):
        raise ValueError("Demonstrations and validation inputs must be disjoint")
    return demos, groups["validation"]




def load_hf_token():
    # Kaggle always retrieves the enabled Secret, never a pasted notebook value.
    if Path("/kaggle").exists():
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            raise RuntimeError("Enable the Kaggle Secret named HF_TOKEN") from None
        if not token:
            raise RuntimeError("The Kaggle HF_TOKEN Secret is empty")
        os.environ["HF_TOKEN"] = token
    return bool(os.environ.get("HF_TOKEN"))


def write_review(output_dir):
    paths = [output_dir / (model.split("/")[-1] + ".json") for model in MODEL_IDS]
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths if p.exists()]
    if not reports:
        return
    lines = ["# Few-shot claim and style review", "",
             "Compare the same source across models. Sentence boundaries are review aids;",
             "check every qualification and causal relationship. All ratings remain blank.", "",
             "Style rubric (0 absent, 1 partial, 2 clear): precise economic vocabulary;",
             "analytical/conditional structure; concise prose resembling the demonstrations.",
             "Fluent or technical wording alone does not establish author style.", ""]
    for initial in reports[0]["results"]:
        key = initial["source_ids"]
        lines.extend([f"## {', '.join(key)}", "", "Source claims:", ""])
        claims = re.split(r"(?<=[.!?])\s+(?=[A-Z])", initial["source"])
        lines.extend(f"{i}. {claim}" for i, claim in enumerate(claims, 1))
        lines.extend(["", "Reference (not shown to either model):", "", initial["reference"], ""])
        for report in reports:
            matches = [r for r in report["results"] if r["source_ids"] == key]
            if not matches:
                continue
            row = matches[0]
            lines.extend([f"### {report['model_id']}", "", row["output"], ""])
            lines.extend(f"- Claim {i}: [ ] preserved [ ] changed [ ] omitted"
                         for i in range(1, len(claims) + 1))
            lines.extend(["- Unsupported additions: ", "- Qualifications or actors changed: ",
                          "- Economic vocabulary (0–2): ", "- Analytical structure (0–2): ",
                          "- Concision and demonstrated style (0–2): ", ""])
    (output_dir / "claim_style_review.md").write_text("\n".join(lines), encoding="utf-8")


def run_training(demos, validation, model_id, dataset_path, output_dir, access_only=False,
                 baseline=None):
    """Inference only; ML imports stay inside this GPU entry point."""
    dual_gpu = model_id == MODEL_IDS[2]
    if not dual_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    import torch
    import transformers
    from transformers import (AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                              BitsAndBytesConfig, Gemma3ForConditionalGeneration,
                              LogitsProcessor, LogitsProcessorList)

    if not torch.cuda.is_available():
        raise RuntimeError("Select a Kaggle T4 GPU")
    if dual_gpu and torch.cuda.device_count() != 2:
        raise RuntimeError("Select GPU T4 x2 and restart the session: exactly two visible GPUs are required")
    if not load_hf_token():
        raise RuntimeError("Set HF_TOKEN outside the repository")
    if access_only:
        from huggingface_hub import hf_hub_download
        for name in ((model_id,) if dual_gpu else MODEL_IDS[:2]):
            hf_hub_download(name, "config.json")
            print(f"Confirmed model access: {name}", flush=True)
        return
    # Gemma 3 4B can overflow in FP16 on T4. Keep 4-bit weights but use
    # FP32 compute/unquantized layers for both models to keep this comparison consistent.
    kwargs = {"dtype": torch.float32, "device_map": {"": 0},
              "attn_implementation": "sdpa",
              "quantization_config": BitsAndBytesConfig(
                  load_in_4bit=True, bnb_4bit_quant_type="nf4",
                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float32)}
    if dual_gpu:
        output_dir.mkdir(parents=True, exist_ok=True)
        hardware_path = output_dir / "12b_hardware_preflight.json"
        hardware = {"model_id": model_id, "status": "checking_memory", "devices": [
            {"device": i, "name": torch.cuda.get_device_name(i),
             "free_bytes": torch.cuda.mem_get_info(i)[0],
             "total_bytes": torch.cuda.mem_get_info(i)[1]} for i in (0, 1)]}
        hardware_path.write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
        # Reserve space for CUDA, activations, generation logits, and the KV cache.
        memory_limits = {}
        for index in (0, 1):
            free, total = torch.cuda.mem_get_info(index)
            budget = min(free - (3 if index == 0 else 2) * 1024**3,
                         total - 3 * 1024**3)
            if budget < 8 * 1024**3:
                raise RuntimeError("Insufficient free GPU memory; start a fresh T4 x2 session")
            memory_limits[index] = budget
        kwargs.update(device_map="balanced", max_memory={**memory_limits, "cpu": 0})
        hardware.update(status="loading_model", gpu_weight_budgets_bytes=memory_limits)
        hardware_path.write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
        print("Loading 12B across two GPUs with reserved generation headroom", flush=True)
    multimodal = model_id != MODEL_IDS[0]
    try:
        if multimodal:
            processor = AutoProcessor.from_pretrained(model_id)
            tokenizer = processor.tokenizer
            model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **kwargs)
        else:
            processor = AutoTokenizer.from_pretrained(model_id)
            tokenizer = processor
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if dual_gpu:
            validate_gpu_placement(model.hf_device_map)
    except Exception:
        if dual_gpu:
            hardware["status"] = "loading_or_placement_failed"
            hardware_path.write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
        raise
    model.eval()
    if dual_gpu:
        hardware["status"] = "placement_passed"
        hardware_path.write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
        print("Confirmed GPU-only model placement across devices 0 and 1", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if baseline is not None:
        (output_dir / "gemma-3-4b-it.json").write_text(
            json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = output_dir / (model_id.split("/")[-1] + ".json")
    report = {"model_id": model_id, "complete": False, "training_called": False, "test_rows_used": 0,
              "validation_rows": 12, "demonstrations": demos, "instruction": INSTRUCTION,
              "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
              "transformers_version": transformers.__version__, "torch_version": torch.__version__,
              "model_revision": getattr(model.config, "_commit_hash", None),
              "quantization": "NF4 double quantization, FP32 compute",
              "inference_revision": "fp32_with_generation_validation_v2",
              "generation": {"do_sample": False, "max_new_tokens": 512}, "results": []}

    def memory_snapshot():
        result = []
        for index in range(2 if dual_gpu else 1):
            free, total = torch.cuda.mem_get_info(index)
            result.append({"device": index, "name": torch.cuda.get_device_name(index),
                           "free_bytes": free, "total_bytes": total,
                           "allocated_bytes": torch.cuda.memory_allocated(index),
                           "reserved_bytes": torch.cuda.memory_reserved(index),
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(index)})
        return result

    if dual_gpu:
        report.update(inference_revision="12b_dual_t4_fp32_v1",
                      device_map={key: str(value) for key, value in model.hf_device_map.items()},
                      gpu_weight_budgets_bytes=memory_limits,
                      memory_after_load=memory_snapshot(),
                      baseline_model_id=baseline["model_id"] if baseline else None)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    class FiniteScores(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # Negative infinity may represent a deliberately suppressed token.
            invalid = (torch.isnan(scores).any() | torch.isposinf(scores).any()
                       | ~torch.isfinite(scores).any(dim=-1).all())
            if bool(invalid):
                raise RuntimeError("Invalid generation scores (NaN/positive infinity/no finite token). "
                                   "The comparison cannot be evaluated.")
            return scores

    for index, row in enumerate(validation, 1):
        prompt = make_prompt(demos, row["plain_english"])
        content = [{"type": "text", "text": prompt}] if multimodal else prompt
        encoded = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt")
        input_device = model.get_input_embeddings().weight.device
        inputs = {k: v.to(input_device) for k, v in encoded.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        if input_length + 512 > 4096:
            raise ValueError("Prompt exceeds the experiment's 4096-token budget; no truncation allowed")
        try:
            with torch.inference_mode():
                tokens = model.generate(**inputs, do_sample=False, max_new_tokens=512,
                                        use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                        remove_invalid_values=False,
                                        logits_processor=LogitsProcessorList([FiniteScores()]))
        except RuntimeError:
            report["error"] = "Generation failed; consult notebook log. Report is incomplete."
            if dual_gpu:
                report["memory_at_generation_error"] = memory_snapshot()
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            raise
        new_tokens = tokens[0, input_length:]
        output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        report["results"].append({
            "source_ids": row["source_ids"], "source": row["plain_english"],
            "reference": row["completion"], "output": output,
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens": input_length, "output_tokens": int(new_tokens.numel()),
            "first_generated_token_ids": new_tokens[:16].tolist(),
            "hit_output_limit": new_tokens.numel() == 512 and int(new_tokens[-1]) not in eos,
            "copied_source": normalized(output) == normalized(row["plain_english"]),
            "claim_review_status": "pending_manual_review"})
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not output:
            raise RuntimeError("Generation decoded to empty text; stopped before processing more examples. "
                               "Report is incomplete and includes token IDs for diagnosis.")
        if dual_gpu and index == 1:
            report["first_generation_check"] = {
                "passed": False, "source_ids": row["source_ids"], "memory": memory_snapshot()}
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            validate_probe(report["results"][-1])
            report["first_generation_check"]["passed"] = True
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print("First-generation check passed; continuing with the remaining 11 validation inputs", flush=True)
        write_review(output_dir)
        print(f"{model_id}: saved validation generation {index}/12", flush=True)
    validate_completion(report)
    report["complete"] = True
    if dual_gpu:
        report["memory_after_all_generations"] = memory_snapshot()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved report to {report_path}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-id", choices=MODEL_IDS, default=MODEL_IDS[0])
    parser.add_argument("--output-dir", default="/kaggle/working/few_shot_comparison")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--baseline", help="completed FP32 4B comparison JSON (required for 12B)")
    parser.add_argument("--check-access", action="store_true",
                        help="check GPU, Secret, and both model licenses before downloading weights")
    args = parser.parse_args(argv)
    dataset_path = Path(args.dataset)
    demos, validation = load_experiment(dataset_path)
    baseline = None
    if args.model_id == MODEL_IDS[2]:
        if not args.baseline:
            parser.error("12B requires --baseline pointing to the saved 4B version 2 JSON")
        baseline = validate_baseline(args.baseline, dataset_path, demos, validation)
    print("Validated 3 reviewed training demonstrations and 12 validation inputs; test rows excluded")
    if args.check:
        print("Dry-run OK. No model downloaded or trained.")
        return
    run_training(demos, validation, args.model_id, dataset_path, Path(args.output_dir), args.check_access,
                 baseline)


if __name__ == "__main__":
    main()
