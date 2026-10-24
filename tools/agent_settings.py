#!/usr/bin/env python3
"""Write the nightly agent's permissions for this machine to .claude/settings.local.json (gitignored).

The agent runs headless in auto permission mode; these rules allow exactly the commands and
files bench/NIGHTLY.md, NIGHTLY_VLLM.md and EXPERIMENT.md need, with this machine's absolute paths from
local.toml, and deny touching the production config. Run it again after changing local.toml.

Usage: tools/agent_settings.py [--print]
"""
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import config  # noqa: E402


def rules():
    serving, models = config.serving(), config.models_dir()
    serve = shlex.join(config.serve())
    stop, ensure = shlex.join(config.serving_cmd("production", "stop")), shlex.join(config.serving_cmd("production", "ensure"))
    prod = config.get("serving", "production_model")
    build = serving / config.get("serving", "candidate_llama_cpp")
    a = lambda p: "/" + str(p)  # "//abs/path": an absolute path in permission rules
    allow = [
        f"Bash({serve} *)", f"Bash(VLLM_EXTRA_ENV=* {serve} *)",
        f"Bash({stop})", f"Bash({ensure}*)", f"Bash({config.get('production', 'start')}*)",
        f"Bash({config.get('production', 'start')} || {ensure}*)",
        "Bash(bench/bench.py *)", "Bash(python3 bench/bench.py *)", "Bash(python3 tools/results_table.py*)",
        f"Bash({sys.executable} {ROOT / 'pipeline' / 'candidates.py'} *)",
        f"Bash({sys.executable} {ROOT / 'pipeline' / 'experiments.py'} *)",
        "Bash(nvidia-smi*)", "Bash(curl -s 127.0.0.1:*)", "Bash(curl -s http://127.0.0.1:*)",
        f"Bash(ln -sfn {prod} {serving}/secrets/*)", f"Bash(rm -r {models}/*)", "Bash(hf download *)",
        f"Bash(cd {build} && git pull && cmake --build build -j*)",
        f"Bash(docker stop {config.get('serving', 'vllm_container')})",
        "Bash(git status*)", "Bash(git diff*)", "Bash(git log*)", "Bash(git add *)", "Bash(git commit *)", "Bash(git push)",
        f"Bash(git -C {serving} *)",
        f"Edit({a(serving)}/models.d/**)", f"Write({a(serving)}/models.d/**)",
        "Edit(JOURNAL.md)", "Edit(README.md)",
    ]
    prod_conf = a(serving / config.get("serving", "production_config"))
    deny = [f"Edit({prod_conf})", f"Write({prod_conf})", "Bash(sudo systemctl stop*)", "Bash(sudo systemctl disable*)",
            "Bash(git push --force*)", "Bash(git push -f*)", "Bash(git commit --no-verify*)", "Bash(git commit -n*)"]
    return {"permissions": {"allow": allow, "deny": deny}}


def main():
    text = json.dumps(rules(), indent=2)
    if "--print" in sys.argv:
        print(text)
        return
    out = ROOT / ".claude" / "settings.local.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")
    print(f"agent_settings: {len(rules()['permissions']['allow'])} allow rules -> {out}")


if __name__ == "__main__":
    main()
