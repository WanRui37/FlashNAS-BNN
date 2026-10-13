from pathlib import Path
import subprocess
import sys

import pytest

from flashnas.cli import ROOT
from tools.data.architectures import read_architectures


def tool(module, *args):
    return subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True)


def test_maintenance_previews_before_deletion(tmp_path):
    cache = tmp_path / '__pycache__'
    cache.mkdir()
    (cache / 'fixture.pyc').write_text('test')
    assert tool('tools.maintenance.clean_python_cache', '--root', tmp_path).returncode == 0
    assert cache.exists()
    assert tool('tools.maintenance.clean_python_cache', '--root', tmp_path, '--apply').returncode == 0
    assert not cache.exists()
    bad = tmp_path / 'BNN1'; good = tmp_path / 'BNN2'; missing = tmp_path / 'BNN3'
    for path in (bad, good, missing): path.mkdir()
    (bad / 'log.txt').write_text('interrupted')
    (good / 'log.txt').write_text('Best Acc:81.40%')
    assert tool('tools.maintenance.prune_incomplete_runs', '--root', tmp_path).returncode == 0
    assert bad.exists()
    assert tool('tools.maintenance.prune_incomplete_runs', '--root', tmp_path, '--apply').returncode == 0
    assert not bad.exists() and good.exists() and missing.exists()


def test_architecture_merge_and_overwrite_protection(tmp_path):
    first = tmp_path / 'first.txt'; second = tmp_path / 'second.txt'; out = tmp_path / 'out.txt'
    record = '[16, 32]\n[0, 0]\n[2, 2]\n'
    first.write_text(record); second.write_text(record + '[32, 64]\n[0, 1]\n[2, 1]\n')
    args = ('--inputs', first, second, '--output', out)
    assert tool('tools.data.merge_architecture_lists', *args).returncode == 0
    assert len(read_architectures(out)) == 2
    content = out.read_text()
    assert tool('tools.data.merge_architecture_lists', *args).returncode != 0
    assert out.read_text() == content


def test_feature_generation_preserves_measured_label_alignment(tmp_path):
    np = pytest.importorskip('numpy')
    architectures = tmp_path / 'architectures.txt'
    architectures.write_text('[16, 32]\n[0, 0]\n[2, 2]\n')
    log = tmp_path / 'runs/BNN1'; log.mkdir(parents=True)
    (log / 'log.txt').write_text(' * Prec@1 81.40\nBest Acc:81.40%\n')
    out = tmp_path / 'arrays'
    result = tool('tools.data.generate_predictor_features', '--dataset', 'imagenet100',
                  '--architectures', architectures, '--logs', tmp_path / 'runs',
                  '--epochs', '1', '--output', out)
    assert result.returncode == 0, result.stderr
    assert np.load(out / '2-np_best_accuracy.npy').tolist() == pytest.approx([81.40])
    assert np.load(out / '2-np_feature_single_vec.npy').shape == (1, 32, 8)
    assert np.load(out / '2-np_feature_total_vec.npy').shape == (1, 11)


def test_root_has_no_numbered_experiment_scripts():
    assert not list(ROOT.glob('[0-9]-*.py'))
