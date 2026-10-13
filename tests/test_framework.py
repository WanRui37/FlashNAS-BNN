import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from flashnas.cli import ROOT, TASKS, execute, make_plan, read_config


def test_all_example_configs_have_real_entrypoints():
    for path in (ROOT / "configs/examples").glob("*.json"):
        config = read_config(path, [])
        plan = make_plan(config)
        assert (Path(plan["cwd"]) / TASKS[config["task"]][0]).exists()
        result = subprocess.run([sys.executable, str(ROOT / "main.py"), "run", "--config", str(path), "--dry-run"],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    plan = make_plan(read_config(ROOT / "configs/examples/train-stage1-imagenet100.json", []))
    assert "torch.distributed.run" in plan["command"]
    assert "stage1/train/main/distributed_amp.py" not in plan["command"]


def test_config_overrides_and_evaluation_checkpoint():
    config = read_config(ROOT / "configs/examples/search-stage1-imagenet100.json", ["parameters.seed=42"])
    assert config["parameters"]["seed"] == 42
    with pytest.raises(ValueError, match="Unknown override"):
        read_config(ROOT / "configs/examples/search-stage1-imagenet100.json", ["parameters.typo=42"])
    with pytest.raises(ValueError, match="Evaluation requires"):
        make_plan({"task": "evaluate", "parameters": {}})


def test_direct_entrypoint_requires_no_project_installation(tmp_path):
    # -S disables site-packages, including any editable installation.
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "main.py"), "run", "--config",
         str(ROOT / "configs/examples/search-stage1-imagenet100.json"), "--dry-run"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "flashnas/search/stage_one.py" in result.stdout


def test_predictor_split_is_disjoint_and_repeatable(tmp_path):
    torch = pytest.importorskip("torch")
    import numpy as np
    from utils.dataset import get_train_valid_set

    arrays = [np.arange(10, dtype=np.float32), np.ones((10, 32, 8), dtype=np.float32),
              np.ones((10, 11), dtype=np.float32)]
    paths = [tmp_path / f"{i}.npy" for i in range(3)]
    for path, array in zip(paths, arrays):
        np.save(path, array)
    torch.manual_seed(2024)
    train, valid = get_train_valid_set(*paths, "imagenet100", 0.8)
    train_ids = set(train.best_accuracy_torch.flatten().tolist())
    valid_ids = set(valid.best_accuracy_torch.flatten().tolist())
    assert len(train_ids) == 8 and len(valid_ids) == 2
    assert not train_ids & valid_ids
    assert train_ids | valid_ids == set(range(10))
    torch.manual_seed(2024)
    repeat, _ = get_train_valid_set(*paths, "imagenet100", 0.8)
    assert torch.equal(repeat.best_accuracy_torch, train.best_accuracy_torch)


def test_model_candidates_are_independent_and_restore_environment(monkeypatch):
    torch = pytest.importorskip("torch")
    from flashnas.models import build_model

    monkeypatch.setenv("ARCH_ALL", "original caller value")
    # Small synthetic architectures verify mechanics, not paper accuracy or latency.
    for channels in ([16, 16, 32], [16, 32, 64]):
        model = build_model(channels + [2, 1, 2] + [0, 0, 0]).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 32, 32))
        assert output.shape == (1, 100)
        assert model.fc.in_features == channels[-1]
    import os
    assert os.environ["ARCH_ALL"] == "original caller value"


def test_missing_artifacts_fail_before_creating_run_directory(tmp_path):
    config = read_config(ROOT / "configs/examples/search-stage1-imagenet100.json", [])
    config["parameters"]["model_path"] = str(tmp_path / "missing.pth")
    config["run_dir"] = str(tmp_path / "run")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "run", "--config", str(path)],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Required artifact not found" in result.stderr
    assert not (tmp_path / "run").exists()


def test_execution_records_failures_and_preserves_existing_runs(tmp_path):
    checkpoint = tmp_path / "input.pth"
    checkpoint.write_text("fixture used only by preflight")
    config = {"task": "plot", "run_dir": str(tmp_path / "run"),
              "parameters": {"checkpoint": str(checkpoint)}}
    plan = {"command": [sys.executable, "-c", "print('fixture output'); raise SystemExit(7)"],
            "cwd": str(ROOT), "environment": {}}
    assert execute(config, plan) == 7
    record = json.loads((tmp_path / "run/run.json").read_text())
    assert record["exit_code"] == 7
    assert record["plan"] == plan
    assert record["git_commit"]
    assert "fixture output" in (tmp_path / "run/console.log").read_text()
    with pytest.raises(ValueError, match="Run already exists"):
        execute(config, plan)


def test_stage2_cpu_timing_runs_through_launcher(tmp_path):
    torch = pytest.importorskip("torch")
    # Synthetic test candidates exercise the pipeline, never manuscript metrics.
    candidates = [[16, 16, 32] + [2, 1, 2] + [0, 0, 0],
                  [16, 32, 64] + [2, 1, 2] + [0, 0, 0]]
    checkpoint = tmp_path / "candidates.pth"
    torch.save({"final_candidates": candidates, "final_candidates_acc": [0.7, 0.8],
                "final_candidates_ops": [100.0, 200.0], "final_candidates_parm": [10.0, 20.0]}, checkpoint)
    config = read_config(ROOT / "configs/examples/search-stage2-pytorch.json", [])
    config["run_dir"] = str(tmp_path / "run")
    config["environment"] = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    config["parameters"].update(use_pth_name=str(checkpoint), logdir=str(tmp_path / "output"),
                                 search_width=32, latency_warmup=1, latency_iterations=2)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "run", "--config", str(path)],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    saved = torch.load(tmp_path / "output/info_second.pth", map_location="cpu")
    assert len(saved["final_candidates_latency"]) == 2
    assert all(value > 0 for value in saved["final_candidates_latency"])
    # Sorting by OP count must not drop a candidate or misalign the accuracy array.
    assert saved["final_candidates"].tolist() == [candidates[1], candidates[0]]
    assert torch.allclose(saved["final_candidates_acc"], torch.tensor([0.8, 0.7]))


def test_python_sources_parse():
    for path in ROOT.rglob("*.py"):
        if not any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            ast.parse(path.read_text(), filename=str(path))


def test_initialization_preserves_boundary_candidates_without_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from tools.data.architectures import read_architectures

    outputs = []
    for trial in range(2):
        config = read_config(ROOT / "configs/examples/init-imagenet100.json", [])
        config["run_dir"] = str(tmp_path / f"run-{trial}")
        output = tmp_path / f"architectures-{trial}.txt"
        config["parameters"].update(logdir=str(tmp_path / f"logs-{trial}"),
                                     init_output_path=str(output), model_path=str(tmp_path / "missing.pth"),
                                     init_max_candidates=2)
        config["parameters"]["population-num"] = 4
        path = tmp_path / f"config-{trial}.json"
        path.write_text(json.dumps(config))
        result = subprocess.run([sys.executable, str(ROOT / "main.py"), "run", "--config", str(path)],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        records = read_architectures(output)
        assert len(records) == len(set(records)) == 6
        # Preserve the intentionally oversized all-3x3 fitting boundary sample.
        assert len(records[1][0]) == 32 and records[1][0][-1] == 1024
        assert all(kernel == 0 for kernel in records[1][1])
        assert "Top-1 acc:" not in result.stdout
        outputs.append(output.read_text())
    assert outputs[0] == outputs[1]
