import argparse
import logging

from .config import load_config
from .data import load_panel, resolve_feature_schema, validate_panel
from .runner import run
from .synthetic import make_synthetic


def main():
    parser = argparse.ArgumentParser(prog="gkx")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--models")
    validate_parser = sub.add_parser("validate-data")
    validate_parser.add_argument("--config", required=True)
    synthetic_parser = sub.add_parser("make-synthetic")
    synthetic_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "make-synthetic":
        make_synthetic(args.output)
        print(args.output)
        return

    cfg = load_config(args.config)
    if args.cmd == "validate-data":
        df = load_panel(cfg["data"], cfg["features"])
        schema = resolve_feature_schema(
            df,
            cfg["features"],
            date_col=cfg["data"]["date_col"],
            industry_cutoff=cfg["split"].get("initial_validation_end"),
        )
        validate_panel(df, cfg["data"], cfg["features"], schema)
        print(
            {
                "rows": len(df),
                "dates": df[cfg["data"]["date_col"]].nunique(),
                "features": len(schema.feature_names),
            }
        )
        return

    out = run(cfg, args.models.split(",") if args.models else None)
    print(out)


if __name__ == "__main__":
    main()
