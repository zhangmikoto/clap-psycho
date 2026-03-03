"""Convert parquet parts to CSV for readability.

We use DuckDB to read parquet without requiring pyarrow/fastparquet.

Example:
  python scripts/parquet_parts_to_csv.py \
    --parts-dir data/clotho_3490684/parts \
    --out-dir  data/clotho_3490684/csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True, help="Directory containing part-*.parquet")
    ap.add_argument("--out-dir", required=True, help="Directory to write part-*.csv")
    ap.add_argument("--pattern", default="part-*.parquet")
    args = ap.parse_args()

    parts_dir = Path(args.parts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parts_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No parquet files found in {parts_dir} matching {args.pattern}")

    con = duckdb.connect(database=":memory:")

    for f in files:
        out_csv = out_dir / (f.stem + ".csv")
        # DuckDB COPY reads parquet directly.
        # DuckDB's COPY does not accept parameter placeholders here; build a safe-ish literal.
        in_path = str(f).replace("'", "")
        out_path = str(out_csv).replace("'", "")
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{in_path}')) TO '{out_path}' (HEADER, DELIMITER ',');"
        )
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
