import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Write a metadata JSON file with GPU architecture information, "
        "reading from a source file and writing to a destination."
    )
    parser.add_argument(
        "input",
        help="Path to the source metadata JSON file to read from.",
    )
    parser.add_argument(
        "destination",
        help="Path to write the output metadata JSON file to.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["cuda", "rocm"],
        help="GPU backend type.",
    )
    parser.add_argument(
        "--archs",
        required=True,
        help="Semicolon-separated list of GPU architectures/capabilities.",
    )
    args = parser.parse_args()

    archs = sorted(set(a for a in args.archs.split(";") if a))

    try:
        with open(args.input) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: input metadata file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: failed to parse input metadata JSON: {e}", file=sys.stderr)
        sys.exit(1)

    data["backend"]["archs"] = archs

    with open(args.destination, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
