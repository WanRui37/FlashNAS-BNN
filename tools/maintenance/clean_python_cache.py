"""List or remove Python bytecode-cache directories."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--apply', action='store_true', help='Remove listed caches; default is preview')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    for cache in sorted(root.rglob('__pycache__')):
        if cache.is_symlink() or not cache.is_dir():
            continue
        print(cache)
        if args.apply:
            shutil.rmtree(cache)


if __name__ == '__main__':
    main()
