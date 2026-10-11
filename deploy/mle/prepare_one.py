"""Prepare one MLE-Bench competition, then park it on the NAS and free the local copy.

Two deviations from `mlebench prepare`, both forced by this deployment:

The archive is fetched with a Bearer token rather than the `kaggle` client. Kaggle
has moved to OAuth 2.0 and its UI now issues only `KG...` access tokens, while the
client MLE-Bench depends on (1.6.17) authenticates with HTTP Basic over a username
and a 32-hex key. The token works perfectly against the REST API; only the client
cannot use it. Nothing downstream changes: the archive is placed exactly where
`prepare` expects it and MLE-Bench's own `checksums.yaml` still verifies the bytes.

`keep_raw` is false by default here. The raw tree and the archive together are
roughly twice the prepared size, and this cluster's local disk is shared.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

DOWNLOAD = "https://www.kaggle.com/api/v1/competitions/data/download-all/{competition}"
INTROSPECT = "https://www.kaggle.com/api/v1/oauth2/introspect"


def read_token(path: Path) -> str:
    token = path.read_text().strip()
    if not token:
        raise SystemExit(f"{path} is empty")
    return token


def check_token(token: str) -> dict:
    """Confirm the token is live before spending time on a download.

    The introspection record carries no `exp`, and a token minted on 8 September was
    still active on the 18th, so these tokens do not expire on a short clock. What the
    check is really for is revocation: a token the user has regenerated on the website
    dies silently mid-download and leaves a truncated archive that only fails later,
    at the checksum.
    """
    request = urllib.request.Request(
        INTROSPECT, data=json.dumps({"token": token}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        record = json.load(response)
    if not record.get("active"):
        raise SystemExit("Kaggle token is not active; generate a new one on the website")
    return record


def download(competition: str, token: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DOWNLOAD.format(competition=competition),
                                     headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        content_type = response.headers.get("Content-Type", "")
        if "zip" not in content_type:
            raise SystemExit(f"{competition}: expected a zip, got {content_type!r}")
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("competition")
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".kaggle/access_token")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/mle-bench/data",
                        help="where mlebench expects <competition>/raw and /prepared")
    parser.add_argument("--nas", type=Path, default=Path("data/mlebench/prepared"))
    parser.add_argument("--keep-raw", action="store_true",
                        help="keep the archive and raw tree; off by default to save local disk")
    parser.add_argument("--skip-nas", action="store_true")
    args = parser.parse_args()

    token = read_token(args.token_file)
    identity = check_token(token)
    competition_dir = args.cache / args.competition
    prepared = competition_dir / "prepared"
    # `prepared/public` and `prepared/private` are created before anything is written
    # into them, so their mere existence means nothing: a run that died at the download
    # leaves both behind empty. Only actual files count as a prepared dataset.
    if prepared.is_dir() and any(f.is_file() for f in prepared.rglob("*")):
        print(json.dumps({"competition": args.competition, "prepared": str(prepared), "cached": True}))
        return 0

    archive = competition_dir / f"{args.competition}.zip"
    if not archive.is_file():
        print(f"downloading {args.competition} as {identity.get('username')} ...", flush=True)
        download(args.competition, token, archive)
    print(f"archive {archive} {archive.stat().st_size} bytes", flush=True)

    # mlebench verifies this archive against its own checksums.yaml, which doubles as
    # proof that the Bearer download produced the same bytes the client would have.
    #
    # `download_dataset` authenticates the old client unconditionally, with no
    # skip-if-present branch, so it is replaced by one that hands back the archive
    # already fetched above. Everything after it -- checksum, extraction, the
    # competition's own prepare_fn -- runs untouched.
    import mlebench.data as mle_data
    from mlebench.registry import registry

    def _use_downloaded_archive(competition_id, download_dir, quiet=False, force=False):
        path = Path(download_dir) / f"{competition_id}.zip"
        if not path.is_file():
            raise SystemExit(f"expected the pre-fetched archive at {path}")
        return path

    mle_data.download_dataset = _use_downloaded_archive
    competition = registry.set_data_dir(args.cache).get_competition(args.competition)
    mle_data.download_and_prepare_dataset(competition, keep_raw=args.keep_raw)

    moved = None
    if not args.skip_nas:
        target = args.nas / args.competition
        target.parent.mkdir(parents=True, exist_ok=True)
        # rsync, not copytree: this NAS is an NFS mount where each file creation costs a
        # round trip, and a per-file copy of an image competition runs an order of
        # magnitude slower than one rsync stream. `--delete` makes the push idempotent
        # so a retry after a half-finished copy converges instead of merging two trees.
        if shutil.which("rsync"):
            subprocess.run(["rsync", "-a", "--delete", f"{prepared}/", f"{target}/"], check=True)
        else:
            # The agent image carries no rsync. Falling back keeps this script usable
            # standalone; the batch driver instead passes --skip-nas and rsyncs from
            # the host, where rsync exists and the slow path is worth avoiding.
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(prepared, target)
        moved = str(target)
    if not args.keep_raw:
        archive.unlink(missing_ok=True)
        shutil.rmtree(competition_dir / "raw", ignore_errors=True)

    sizes = {name: sum(f.stat().st_size for f in (prepared / name).rglob("*") if f.is_file())
             for name in ("public", "private") if (prepared / name).is_dir()}
    print(json.dumps({"competition": args.competition, "prepared": str(prepared),
                      "nas": moved, "bytes": sizes, "raw_kept": args.keep_raw}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
