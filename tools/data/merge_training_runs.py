"""Copy legacy BNN run directories into one directory with distinct numeric IDs."""
import argparse
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for source in args.sources:
        if not source.is_dir():
            parser.error(f'Source directory does not exist: {source}')
        runs.extend(sorted((p for p in source.iterdir() if p.is_dir() and not p.is_symlink()
                            and re.fullmatch(r'BNN\d+', p.name)), key=lambda p: int(p.name[3:])))
    if any(source.resolve() == args.output.resolve() or source.resolve() in args.output.resolve().parents
           for source in args.sources):
        parser.error('Output must be outside source directories')
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'source-map.txt').open('w') as manifest:
        for index, source in enumerate(runs, 1):
            target = args.output / f'BNN{index}'
            shutil.copytree(source, target)
            manifest.write(f'{target.name}\t{source.resolve()}\n')
    print(f'Copied {len(runs)} runs to {args.output}')


if __name__ == '__main__':
    main()
