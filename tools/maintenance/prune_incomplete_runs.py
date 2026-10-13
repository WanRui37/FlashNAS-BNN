"""List legacy BNN runs whose log has no completed Best Acc record."""
import argparse
from pathlib import Path
import shutil


def incomplete_runs(root):
    for run in sorted(root.glob('BNN*')):
        if not run.is_dir() or run.is_symlink():
            continue
        log = run / 'log.txt'
        if log.is_file() and 'Best Acc:' not in log.read_text():
            yield run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Delete listed directories; default is preview')
    args = parser.parse_args()
    for run in incomplete_runs(args.root.resolve(strict=True)):
        print(run)
        if args.apply:
            shutil.rmtree(run)


if __name__ == '__main__':
    main()
