import os
import logging
import argparse
import json

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGGER = logging.getLogger(__name__)


def test_os():
    print(base_dir)


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Defining command line input "--config-file"
    parser.add_argument("--config-file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    with open(args.config_file) as f:
        config = json.load(f)

    print(config)
