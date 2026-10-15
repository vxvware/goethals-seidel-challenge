import subprocess
import sys
import pytest
from verify.common import ROOT, read_json
from verify.check_submission_scope import check_changes
from verify.check_authorship import check_author


def test_scope_separates_infrastructure():
    check_changes([("A", "quadruples/new.json"), ("A", "evidence/new.json"), ("A", "notes/new.md")])
    for changes in (
        [("M", "quadruples/old.json")],
        [("A", "quadruples/new.json"), ("M", "verify/gs_verify.py")],
        [("A", "quadruples/new.json"), ("A", "evidence/engine.py")],
        [("A", "quadruples/new.json"), ("M", ".github/workflows/ci.yml")],
    ):
        with pytest.raises(ValueError):
            check_changes(changes)


def test_authorship_uses_submitter():
    check_author({"id": "a", "submitted_by": "Trevor-Vincent"}, "trevor-vincent")
    with pytest.raises(ValueError):
        check_author({"id": "a", "submitted_by": "another"}, "trevor-vincent")


def test_cli_rejects_duplicate_without_writing():
    path = next((ROOT / "quadruples").glob("*.json"))
    run = subprocess.run(
        [sys.executable, str(ROOT / "tools/gs"), "submit", str(path)],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1 and "equivalent entry" in run.stderr


def test_json_duplicate_keys_rejected(tmp_path):
    p = tmp_path / "duplicate.json"
    p.write_text('{"n":3,"n":5}')
    with pytest.raises(ValueError):
        read_json(p)
