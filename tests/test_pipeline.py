from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PipelineSmokeTests(unittest.TestCase):
    def run_script(self, script: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_prepare_documents_emits_stable_provenance_chunks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs"
            docs.mkdir()
            (docs / "guide.md").write_text("alpha beta gamma delta epsilon zeta eta theta", encoding="utf-8")
            output = root / "chunks.jsonl"
            result = self.run_script(
                "prepare_documents.py", "--input", str(docs), "--output", str(output),
                "--chunk-words", "5", "--overlap-words", "1",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["source"], "guide.md")
            self.assertEqual(rows[0]["location"], "document, chunk 1")
            self.assertNotEqual(rows[0]["chunk_id"], rows[1]["chunk_id"])

    def test_validate_dataset_and_duplicate_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            record = {"messages": [
                {"role": "system", "content": "Be grounded."},
                {"role": "user", "content": "Explain the stated method."},
                {"role": "assistant", "content": "The source describes the method clearly."},
            ]}
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            result = self.run_script("validate_dataset.py", str(path))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["examples"], 2)
            self.assertEqual(report["duplicate_examples"], 1)
            strict = self.run_script("validate_dataset.py", str(path), "--fail-on-duplicates")
            self.assertEqual(strict.returncode, 1)

    def test_invalid_role_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.jsonl"
            row = {"messages": [
                {"role": "human", "content": "Question?"},
                {"role": "assistant", "content": "A sufficiently long answer."},
            ]}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = self.run_script("validate_dataset.py", str(path))
            self.assertEqual(result.returncode, 1)
            self.assertIn("invalid role", result.stdout)

    def test_trainer_preflight_does_not_require_ml_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            row = {"messages": [
                {"role": "user", "content": "What does the source state?"},
                {"role": "assistant", "content": "It states the answer in the cited section."},
            ]}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = self.run_script("train_unsloth.py", "--train-file", str(path), "--preflight-only")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["train_examples"], 1)
            self.assertIn("Qwen2.5-Coder", report["model"])

    def test_teacher_json_parser_accepts_wrapped_examples(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            from distill_dataset import extract_examples
        finally:
            sys.path.pop(0)
        parsed = extract_examples('```json\n{"examples":[{"instruction":"Explain this concept?","response":"A grounded answer."}]}\n```')
        self.assertEqual(parsed[0]["instruction"], "Explain this concept?")


if __name__ == "__main__":
    unittest.main()
