import argparse
import sys


def create_parser() -> argparse.ArgumentParser:
    """
    Initialize the command line argument parser.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", default="input/",
                        help="Path to directory with input data.")
    parser.add_argument("-o", "--output", default="output/",
                        help="Directory where to store results.")
    parser.add_argument("-r", "--random", default=42, type=int, help="Random seed.")
    parser.add_argument("-n", "--number", default=3, type=int, help="Number of nearest neighbors")
    parser.add_argument("-l", "--language", default="en_core_web_lg", type=str,
                        help="Default value to load spacy.")

    return parser


def main():
    """Entry point."""
    parser = create_parser()
    args = parser.parse_args()
    print(args)
    # TODO: import & call functionalty here


if __name__ == "__main__":
    sys.exit(main())
