"""Static contract tests between the Colab notebook and Python outputs."""
import ast
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _notebook_code():
    notebook = json.loads((ROOT / "NeuroGuard_AI_Final.ipynb").read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _dict_keys(node):
    if not isinstance(node, ast.Dict):
        return set()
    return {
        key.value
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


class NotebookContractTests(unittest.TestCase):
    def test_notebook_audit_fields_exist_in_discover_recordings(self):
        core = ast.parse((ROOT / "neuroguard_core.py").read_text(encoding="utf-8"))
        audit_keys = set()
        for function in (n for n in ast.walk(core)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and n.name == "discover_recordings"):
            for node in ast.walk(function):
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "audit"
                    for target in node.targets
                ):
                    audit_keys.update(_dict_keys(node.value))
                elif (isinstance(node, ast.AnnAssign)
                      and isinstance(node.target, ast.Name)
                      and node.target.id == "audit"):
                    audit_keys.update(_dict_keys(node.value))
        used_keys = set(re.findall(r"\baudit\[['\"]([^'\"]+)['\"]\]", _notebook_code()))
        self.assertTrue(audit_keys, "Could not locate audit schema in discover_recordings")
        self.assertTrue(used_keys, "Could not locate audit field references in notebook")
        self.assertEqual(
            used_keys - audit_keys,
            set(),
            f"Notebook references audit fields not produced by discover_recordings: {sorted(used_keys - audit_keys)}",
        )

    def test_notebook_test_case_field_exists_in_saved_manifest(self):
        training = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
        manifest_keys = set()
        for call in (n for n in ast.walk(training)
                     if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name)
                     and n.func.id == "write_json"
                     and len(n.args) >= 2):
            first_arg = ast.unparse(call.args[0])
            if "manifest.json" in first_arg:
                manifest_keys.update(_dict_keys(call.args[1]))
        notebook_code = _notebook_code()
        accessed_keys = set(re.findall(
            r"detector\.manifest\.get\(['\"]([^'\"]+)['\"]", notebook_code
        ))
        self.assertTrue(manifest_keys, "Could not locate saved model manifest schema")
        self.assertTrue(accessed_keys, "Could not locate model manifest accesses in notebook")
        self.assertEqual(
            accessed_keys - manifest_keys,
            set(),
            f"Notebook accesses manifest fields not saved by train.py: {sorted(accessed_keys - manifest_keys)}",
        )


    def test_generator_reproduces_published_notebook(self):
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            generated = Path(temp_dir) / "NeuroGuard_AI_Final.ipynb"
            subprocess.run(
                [sys.executable, str(ROOT / "make_notebook.py"), "--output", str(generated)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            published = json.loads(
                (ROOT / "NeuroGuard_AI_Final.ipynb").read_text(encoding="utf-8")
            )
            rebuilt = json.loads(generated.read_text(encoding="utf-8"))
            self.assertEqual(
                rebuilt,
                published,
                "make_notebook.py output differs from the checked-in Colab notebook",
            )


if __name__ == "__main__":
    unittest.main()
