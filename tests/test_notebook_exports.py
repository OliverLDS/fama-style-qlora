"""Check generated Kaggle modules without executing notebook jobs or loading ML."""
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NotebookExportTests(unittest.TestCase):
    def test_cells_compile_and_exports_run_without_project_package(self):
        notebooks = sorted((ROOT / "kaggle").glob("*.ipynb"))
        self.assertTrue(notebooks)
        for path in notebooks:
            with self.subTest(notebook=path.name), tempfile.TemporaryDirectory() as folder:
                exports = []
                for cell in json.loads(path.read_text())["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    if source.startswith("%pip "):
                        continue
                    if source.startswith("%%writefile "):
                        header, source = source.split("\n", 1)
                        target = Path(folder) / Path(header.split(maxsplit=1)[1]).name
                        target.write_text(source)
                        exports.append(target)
                    tree = ast.parse(source, filename=path.name)
                    compile(tree, path.name, "exec")
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom):
                            self.assertFalse((node.module or "").startswith("fama_style"))
                env = dict(os.environ)
                env.pop("PYTHONPATH", None)
                for exported in exports:
                    result = subprocess.run(
                        [sys.executable, str(exported), "--help"], cwd=folder,
                        env=env, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
