#!/usr/bin/env python
"""Phase 0 — Data Acquisition.

Pulls MedEmbed triplets and NFCorpus once, asserts the spec's row counts, and
freezes artifacts/raw/ read-only.

    python scripts/00_download.py [--force] [--no-freeze]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lyra_capstone import manifest, paths, seeds  # noqa: E402
from lyra_capstone.config import DataConfig, load_config  # noqa: E402
from lyra_capstone.data import load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=paths.CONFIGS / "data.yaml")
    ap.add_argument("--force", action="store_true", help="re-download over a frozen raw tree")
    ap.add_argument("--no-freeze", action="store_true", help="skip chmod -R a-w")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )

    cfg = load_config(DataConfig, args.config)
    seed = seeds.set_seed(cfg.seed)
    paths.ensure_dirs()

    medembed = load.download_medembed(paths.RAW_MEDEMBED, cfg.medembed, force=args.force)
    nfcorpus = load.download_nfcorpus(paths.RAW_NFCORPUS, cfg.nfcorpus, force=args.force)

    manifest.write_manifest(
        paths.RAW,
        phase="00_download",
        config=cfg,
        seed=seed,
        outputs=[paths.RAW_MEDEMBED, paths.RAW_NFCORPUS],
        extra={"medembed": medembed, "nfcorpus": nfcorpus},
    )

    if not args.no_freeze:
        load.set_readonly(paths.RAW_MEDEMBED)
        load.set_readonly(paths.RAW_NFCORPUS)
        logging.info("froze %s read-only", paths.RAW)

    print("\nPhase 0 complete.")
    print(f"  MedEmbed  {medembed['counts']} @ {medembed['revision'][:12]}")
    print(f"  NFCorpus  {nfcorpus['counts']} @ {nfcorpus['revision'][:12]}")
    print(f"  manifest  {paths.RAW / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
