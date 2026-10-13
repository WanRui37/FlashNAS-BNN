"""Dependency-light experiment launcher; research dependencies load in child processes."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "predictor": ("net_predict/3-train_acc_predict.py", "."),
    "search-stage1": ("1-search_first_stage.py", "."),
    "search-stage2": ("1-search_second_stage.py", "."),
    "train": ("train/main/distributed_amp.py", "train_imagenet"),
    "evaluate": ("train/main/distributed_amp.py", "train_imagenet"),
    "plot": ("1-search_draw_parato.py", "."),
}


def read_config(path, overrides):
    config = json.loads(Path(path).read_text())
    for override in overrides:
        key, separator, raw = override.partition("=")
        if not separator:
            raise ValueError("Overrides must use KEY=VALUE")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f"Unknown override: {key}")
        node[parts[-1]] = value
    if set(config) - {"task", "parameters", "environment", "run_dir", "nproc"}:
        raise ValueError("Unknown configuration fields")
    if config.get("task") not in TASKS:
        raise ValueError(f"task must be one of {', '.join(TASKS)}")
    if not isinstance(config.get("parameters"), dict):
        raise ValueError("parameters must be an object")
    return config


def make_plan(config):
    script, cwd = TASKS[config["task"]]
    params = dict(config["parameters"])
    if config["task"] == "evaluate":
        params["evaluate"] = True
        if not params.get("resume_model") or not params.get("resume"):
            raise ValueError("Evaluation requires resume=True and resume_model checkpoint")
    command = [sys.executable]
    if config["task"] in ("train", "evaluate"):
        nproc = config.get("nproc", 1)
        if not isinstance(nproc, int) or nproc < 1:
            raise ValueError("nproc must be a positive integer")
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}"]
    command.append(script)
    # Existing parsers distinguish boolean flags from literal True/False options.
    flags = {"evaluate", "save_pth", "nsga2", "fpga", "deterministic", "sync_bn"}
    for key, value in params.items():
        if value is None:
            continue
        if key in flags:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a boolean")
            if value:
                command.append(f"--{key}")
        else:
            command.append(f"--{key}")
            command.extend(str(v) for v in (value if isinstance(value, list) else [value]))
    env = {str(k): str(v) for k, v in config.get("environment", {}).items()}
    return {"command": command, "cwd": str(ROOT / cwd), "environment": env}


def preflight(config, plan):
    p = config["parameters"]
    required = []
    if config["task"] == "predictor":
        required = [Path(p.get("data_root", "2-npy")) / p.get("dataset", "cifar10") / name
                    for name in ("2-np_best_accuracy.npy", "2-np_feature_single_vec.npy", "2-np_feature_total_vec.npy")]
        if p.get("mode") == "test":
            required.append(Path(p["out_path"]) / p["dataset"] / f"CharRNN_{p.get('model_arch', 'OUR')}_best.pth")
    elif config["task"] == "search-stage1":
        required = [Path(p["model_path"])]
    elif config["task"] == "search-stage2":
        required = [Path(p["use_pth_name"])]
        if not (p.get("mode") == "real_time" and p.get("save_pth")):
            required.append(Path(p["model_path"]))
        if p.get("mode") == "predict":
            if p.get("platform") == "gpu":
                raise ValueError("Legacy CLE code implements fpga/cpu branches; GPU CLE is not provided")
            if p.get("platform") == "cpu":
                required.append(Path(p["bw_output_path"]) / "model_best.pth")
        if p.get("mode") == "real_time" and p.get("latency_mode") == "onnx":
            raise ValueError("Legacy ONNX mode reads external timing logs without candidate IDs; use pytorch or aligned use_pth data")
    elif config["task"] in ("train", "evaluate"):
        required = [Path(p["data_path"]) / split for split in ("train", "val")]
        for key in ("baseline_model", "resume_model"):
            if p.get(key):
                required.append(Path(p[key]))
        if not config.get("environment", {}).get("ARCH_ALL"):
            raise ValueError("Training/evaluation requires explicit ARCH_ALL architecture")
        candidate = ast.literal_eval(config["environment"]["ARCH_ALL"])
        from .models import architecture_environment
        with architecture_environment(candidate):
            pass
    elif config["task"] == "plot":
        required = [Path(p["checkpoint"])]
    for path in required:
        resolved = path if path.is_absolute() else Path(plan["cwd"]) / path
        if not resolved.exists():
            raise ValueError(f"Required artifact not found: {resolved}")


def execute(config, plan):
    preflight(config, plan)
    run_dir = ROOT / config["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "run.json"
    if manifest.exists():
        raise ValueError(f"Run already exists: {run_dir}. Set a new run_dir and output paths.")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    record = {"config": config, "plan": plan, "started_at": datetime.now(timezone.utc).isoformat(),
              "git_commit": git.stdout.strip(), "git_dirty": bool(status.stdout),
              "python": sys.version}
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    env = os.environ.copy()
    env.update(plan["environment"])
    # Child scripts work from their original directories, regardless of the caller's cwd.
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        with (run_dir / "console.log").open("w") as log:
            process = subprocess.Popen(plan["command"], cwd=plan["cwd"], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        code = 130
    record.update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description="FlashNAS-BNN configured experiments")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor", help="Report installed research dependencies")
    run = sub.add_parser("run", help="Run a JSON experiment configuration")
    run.add_argument("--config", required=True)
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--dry-run", action="store_true", help="Print command without requiring data or starting jobs")
    args = parser.parse_args(argv)
    if args.action == "doctor":
        names = ("torch", "torchvision", "numpy", "scipy", "sklearn", "timm", "matplotlib", "pandas", "seaborn", "apex", "torch_geometric")
        report = {name: importlib.util.find_spec(name) is not None for name in names}
        print(json.dumps({"python": sys.executable, "root": str(ROOT), "installed": report}, indent=2))
        return 0
    try:
        config = read_config(args.config, args.set)
        plan = make_plan(config)
        print(shlex.join(plan["command"]), flush=True)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        return execute(config, plan)
    except (ValueError, KeyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
