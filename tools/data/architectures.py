"""Read legacy three-line architecture records: channels, kernels, strides."""
import ast
from pathlib import Path


def read_architectures(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if len(lines) % 3:
        raise ValueError('Each architecture must occupy three lines: channels, kernels, strides')
    records = []
    for offset in range(0, len(lines), 3):
        record = tuple(tuple(ast.literal_eval(line)) for line in lines[offset:offset+3])
        channels, kernels, strides = record
        if not channels or not len(channels) == len(kernels) == len(strides):
            raise ValueError(f'Unequal architecture lengths at record {offset // 3 + 1}')
        if any(type(value) is not int for array in record for value in array):
            raise ValueError('Architecture entries must be integers')
        records.append(record)
    return records
