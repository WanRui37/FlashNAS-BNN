"""Combine architecture text files while preserving the first occurrence."""
import argparse
from pathlib import Path
from tools.data.architectures import read_architectures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = dict.fromkeys(record for path in args.inputs for record in read_architectures(path))
    with args.output.open('x') as out:
        for record in records:
            for array in record:
                out.write(str(list(array)) + '\n')
    print(f'Wrote {len(records)} unique architectures to {args.output}')


if __name__ == '__main__':
    main()
