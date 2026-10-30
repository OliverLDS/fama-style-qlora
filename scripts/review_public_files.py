"""Audit tracked/staged publication bytes without printing credential values."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT_FILES = {".gitignore", ".gitattributes", "README.md", "LICENSE", "AGENTS.md",
              "pyproject.toml", "requirements-kaggle.txt", "train.py"}
SYNTHETIC_SHA = "74179c0fe49bb6e36e992456a793b199ee10ffb59c4513092b7059a1ce1456cf"
PATTERNS = {
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "API secret": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "AWS key": re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "home directory": re.compile(rb"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
}


def allowed(path):
    parts = Path(path).parts
    if not parts or any(p in ("__pycache__", ".venv", "node_modules") for p in parts):
        return False
    if path in ROOT_FILES:
        return True
    if parts[0] == "src":
        return path.endswith((".py", ".md")) or path == "src/fama_style/evaluation/templates/source_control_review.html"
    if parts[0] in ("scripts", "tests"):
        return path.endswith(".py") or path in {"scripts/data/extract_comparison_ocr.swift", "scripts/data/extract_author_pages.swift"}
    if parts[0] == "kaggle":
        return len(parts) == 2 and (path.endswith(".ipynb") or parts[-1] == "README.md")
    if parts[0] == "docs":
        return path.endswith((".md", ".json"))
    if parts[0] == ".github":
        return path.endswith((".yml", ".yaml")) and len(parts) == 3 and parts[1] == "workflows"
    return path in ("data/README.md", "data/historical/train.example.jsonl",
                    "data/fama_corpus_v1/rewrite_instruction.txt")


def audit_file(path, content):
    issues = []
    if not allowed(path):
        issues.append("outside public allowlist")
    if len(content) > 2*1024*1024:
        issues.append("unexpected file size above 2 MiB")
    if b"\x00" in content:
        issues.append("binary content")
    for label, pattern in PATTERNS.items():
        if pattern.search(content):
            issues.append(label + " pattern")
    if path == "data/historical/train.example.jsonl" and hashlib.sha256(content).hexdigest() != SYNTHETIC_SHA:
        issues.append("synthetic example contents changed; explicit publication review required")
    if path.endswith(".ipynb"):
        notebook = json.loads(content)
        if any(cell.get("outputs") or cell.get("execution_count") is not None or cell.get("attachments")
               for cell in notebook["cells"]):
            issues.append("saved notebook outputs, execution counts or attachments")
    return issues


def private_paragraphs(root):
    values = set()
    keys = {"original_paragraph", "plain_english", "completion", "original_target", "reviewed_target", "reviewed_source", "text", "raw_ocr", "source_paragraph", "candidate", "source", "target"}
    def visit(value):
        if isinstance(value, dict):
            for key, text in value.items():
                if key in keys and isinstance(text, str) and len(text) >= 150:
                    values.add(" ".join(text.split()))
                else:
                    visit(text)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    for folder in (root / "data", root / "archive", root / "research_runs"):
        for path in folder.rglob("*.jsonl"):
            if path.name == "train.example.jsonl":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    visit(json.loads(line))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-scan", action="store_true", help="Also compare with local-only paper paragraphs")
    args = parser.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    paragraphs = private_paragraphs(root) if args.private_scan else set()
    failures, total = [], 0
    for path in filter(None, files):
        # Inspect index bytes, not possibly different working-tree files.
        content = subprocess.check_output(["git", "show", ":"+path], cwd=root)
        total += len(content)
        issues = audit_file(path, content)
        if paragraphs:
            text = content.decode("utf-8", errors="replace")
            if path.endswith(".ipynb"):
                text = "\n".join("".join(cell["source"]) for cell in json.loads(content)["cells"])
            text = " ".join(text.split())
            if any(paragraph in text for paragraph in paragraphs):
                issues.append("matches a local-only paper paragraph")
        failures.extend({"path": path, "issue": issue} for issue in issues)
    if failures:
        # Never print a matching token, source snippet or private artifact value.
        print(json.dumps({"passed": False, "findings": failures}, indent=2))
        raise SystemExit(1)
    print(json.dumps({"passed": True, "files": len(list(filter(None, files))), "bytes": total,
                      "private_paragraph_comparison": bool(paragraphs)}))


if __name__ == "__main__":
    main()
