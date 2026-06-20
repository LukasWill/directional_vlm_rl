import ast
import importlib.util
import json
from pathlib import Path

import yaml
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "Code"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_assignments():
    tree = ast.parse((CODE / "configs" / "metaworld.py").read_text())
    values = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            if target.value.id == "config":
                try:
                    values[target.attr] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return values


def test_public_configuration_contract():
    config = config_assignments()
    assert config["exp_name"] == "furl"
    assert config["wandb_entity"] == ""
    assert config["pos_alg_version"] in range(11)
    assert config["dir_alg_version"] in range(6)


def test_requirements_are_pip_parseable():
    requirements = CODE / "requirements.txt"
    lines = [
        line.strip()
        for line in requirements.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines
    for line in lines:
        Requirement(line)


def test_environment_file_references_runtime_requirements():
    environment = yaml.safe_load((ROOT / "environment.yml").read_text())
    assert environment["name"] == "directional-vlm-rl"
    pip_entries = next(item["pip"] for item in environment["dependencies"] if isinstance(item, dict))
    assert "-r Code/requirements.txt" in pip_entries


def test_citation_metadata():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text())
    assert citation["cff-version"] == "1.2.0"
    assert citation["repository-code"] == "https://github.com/LukasWill/directional_vlm_rl"
    assert citation["license"] == "MIT"
    assert citation["date-released"] == "2026-06-20"
    assert "preferred-citation" not in citation


def test_saved_run_analysis(tmp_path):
    analyze_runs = load_module("analyze_runs", CODE / "analyze_runs.py")
    run_dir = tmp_path / "runs" / "door-open-v2-goal-hidden"
    run_dir.mkdir(parents=True)
    run = {
        "env": "door-open-v2-goal-hidden",
        "palg": 0,
        "dalg": 3,
        "data": {
            "step": {"0": 100000, "1": 200000},
            "train_success": {"0": 0.25, "1": 0.75},
        },
    }
    (run_dir / "run.json").write_text(json.dumps(run))

    summary = analyze_runs.summarize_release_runs(str(tmp_path / "runs"))
    assert len(summary) == 1
    assert summary.iloc[0]["env"] == "door-open-v2-goal-hidden"
    assert summary.iloc[0]["dir_alg"] == 3


def test_public_tree_contains_no_notebooks():
    assert not list(ROOT.rglob("*.ipynb"))
