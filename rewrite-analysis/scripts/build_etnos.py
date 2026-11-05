"""Write the canonical ETNO samples to data/processed/ and print a summary.

Usage:
    python scripts/build_etnos.py            # from the bundled MPC snapshot
    python scripts/build_etnos.py --sbdb     # from a live (cached) JPL SBDB query
"""

import argparse

from planetix.data import CUTS, load_catalog, select_etnos
from planetix.data.catalog import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sbdb", action="store_true", help="use JPL SBDB instead of the MPC snapshot")
    parser.add_argument("--refresh", action="store_true", help="re-download SBDB data")
    args = parser.parse_args()

    if args.sbdb:
        from planetix.data.sbdb import fetch

        catalog, source = fetch(refresh=args.refresh), "sbdb"
    else:
        catalog, source = load_catalog(), "mpc"

    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(catalog)} orbits in {source} catalogue")
    for name in CUTS:
        loose = select_etnos(catalog, name, min_arc_days=0, reliable_only=False)
        df = select_etnos(catalog, name)
        path = out_dir / f"etnos_{name}_{source}.csv"
        df.to_csv(path, index=False)
        print(f"  {name:9s} {CUTS[name]}: {len(loose):3d} before quality cuts, {len(df):3d} after -> {path.name}")


if __name__ == "__main__":
    main()
