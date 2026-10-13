"""Estimate convolution parameters for a 3x3 + two 1x1 block chain."""
import argparse


def estimate_parameters(channels):
    if not channels or any(c <= 0 for c in channels):
        raise ValueError('Channels must be positive integers')
    return 27 * channels[0] + sum(11 * a * b for a, b in zip(channels, channels[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', nargs='+', type=int, required=True)
    args = parser.parse_args()
    print(estimate_parameters(args.channels))


if __name__ == '__main__':
    main()
