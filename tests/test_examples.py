"""The offline chapter examples (code/chNN_*/*.py) run and print the key numbers.

Scripts that call the DeepSeek API are skipped; the code-review assistant
runs in its offline ``--mock`` mode.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
NEEDS_API = {"ch16_deployment/api_chat.py", "ch17_specialized/code_completion_api.py",
             "ch18_practice/json_mode.py", "ch18_practice/reasoner_chat.py"}
ARGS = {"ch18_practice/code_review_assistant.py": ["sample_buggy.py", "--mock"]}
# key numbers that must appear in the output (reproducibility of the book's figures)
EXPECT = {
    "ch04_mla/mla_numpy.py": ["OK: absorbed MLA equals naive MLA", "x3.2 smaller"],
    "ch05_moe/moe_router.py": ["C(64, 8) = 4,426,165,368"],
    "ch11_parallelism/pipeline_bubble.py": ["formula (p-1)(F+B)=9"],
    "ch13_grpo/grpo.py": ["P(correct) rose from"],
    "ch09_pretraining/lr_schedule.py": ["total optimizer steps"],
    "ch07_mtp/spec_decode_toy.py": ["acceptance"],
}
SCRIPTS = sorted(str(p.relative_to(CODE)) for p in CODE.glob("ch[0-9][0-9]_*/*.py"))


@pytest.mark.parametrize("script", [s for s in SCRIPTS if s not in NEEDS_API])
def test_example_runs(script):
    path = CODE / script
    env = {**os.environ, "DEEPSEEK_API_KEY": "", "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, path.name, *ARGS.get(script, [])], cwd=path.parent,
                       capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    for needle in EXPECT.get(script, []):
        assert needle in r.stdout, f"{needle!r} missing in output of {script}"
