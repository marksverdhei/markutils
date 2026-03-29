import argparse
import sys


def _cmd_convert(args: argparse.Namespace) -> int:
    from markutils.utils.data import load_dataframe, save_dataframe

    try:
        df = load_dataframe(args.input)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        save_dataframe(df, args.output)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Converted {args.input} → {args.output} ({len(df)} rows)")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from markutils.checkpoints import get_state_dict, print_state_dict_shapes

    try:
        tensors = get_state_dict(args.path)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Keys: {len(tensors)}")
    print_state_dict_shapes(tensors)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="markutils",
        description="Utilities for data conversion and model inspection.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # convert subcommand
    p_convert = sub.add_parser(
        "convert",
        help="Convert a data file between supported formats (.csv, .tsv, .json, .jsonl, .parquet).",
    )
    p_convert.add_argument("input", help="Input file path.")
    p_convert.add_argument("output", help="Output file path (format inferred from extension).")

    # inspect subcommand
    p_inspect = sub.add_parser(
        "inspect",
        help="Print tensor shapes from a safetensors checkpoint.",
    )
    p_inspect.add_argument("path", help="Path to a .safetensors file.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "convert":
        sys.exit(_cmd_convert(args))
    elif args.command == "inspect":
        sys.exit(_cmd_inspect(args))


if __name__ == "__main__":
    main()
