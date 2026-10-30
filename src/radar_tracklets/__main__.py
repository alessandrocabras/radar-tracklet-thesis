"""Run one explicitly configured private experiment."""
import argparse
import json
from .experiments import external_path, run_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('configuration', help='External private JSON configuration')
    args = parser.parse_args()
    config = json.loads(external_path(args.configuration).read_text(encoding='utf-8'))
    run_private(config)


if __name__ == '__main__':
    main()
