"""Publish reviewed CIK issuer names, not membership, business segments or coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from market_predictor.catalysts.issuer_events.identity_publication import publish_issuer_identity_authority
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with heavy_job_lease("publish-issuer-identity-authority",
        runtime_dir=inside(root, heavy_job_runtime_dir())):
        print(json.dumps(publish_issuer_identity_authority(root=root, config=root / args.config,
            expected_config_sha256=args.expected_config_sha256, output_directory=root / args.out_dir), sort_keys=True))


if __name__ == "__main__":
    main()
