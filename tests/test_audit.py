import subprocess
import sys


def _write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_audit_runs_on_toy_data(tmp_path):
    raw = tmp_path / "raw"
    rows_a = "id,language,prompt,response,label\n1,sw,hello there friend,ok,safe\n2,sw,do bad thing now,no,unsafe\n"
    rows_b = "id,language,prompt,response,label\n1,sw,hello there friend,ok,Ethical & Moral Risk\n2,sw,do bad thing now,no,Violence & Physical Harm\n"
    for sp in ("train", "dev"):
        _write(raw / "task_a_safety" / "swahili" / f"{sp}.csv", rows_a)
        _write(raw / "task_b_category" / "swahili" / f"{sp}.csv", rows_b)
    test = "id,language,prompt,response\n1,sw,hello again friend,ok\n"
    _write(raw / "task_a_safety" / "swahili" / "test.csv", test)
    _write(raw / "task_b_category" / "swahili" / "test.csv", test)
    r = subprocess.run([sys.executable, "scripts/audit_dataset.py", "--data-dir", str(raw),
                        "--output-dir", str(tmp_path / "o"), "--tables-dir", str(tmp_path / "t"),
                        "--docs-dir", str(tmp_path / "d")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "o" / "dataset_audit.json").exists()
