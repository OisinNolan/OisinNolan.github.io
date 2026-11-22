"""Download the raw Wiktextract dump (English Wiktionary) from kaikki.org.

Resumable: re-running continues a partial download. Checks the final size
against the server's Content-Length.

    python build/fetch.py            # -> raw/raw-wiktextract-data.jsonl.gz
"""
import sys
import urllib.request
from pathlib import Path

URL = "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "raw" / "raw-wiktextract-data.jsonl.gz"
UA = {"User-Agent": "etymap-build/0.1"}


def remote_size() -> int:
    req = urllib.request.Request(URL, method="HEAD", headers=UA)
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size()
    have = DEST.stat().st_size if DEST.exists() else 0
    if have == total:
        print(f"already complete: {DEST} ({total / 1e9:.2f} GB)")
        return
    if have > total:
        sys.exit(f"local file larger than remote ({have} > {total}); delete it and retry")

    req = urllib.request.Request(URL, headers={**UA, "Range": f"bytes={have}-"})
    with urllib.request.urlopen(req) as r, open(DEST, "ab") as f:
        if have and r.status != 206:
            sys.exit("server ignored Range request; delete the partial file and retry")
        done, last = have, -1
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total)
            if pct != last:
                print(f"{pct:3d}%  {done / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
                last = pct
    if DEST.stat().st_size != total:
        sys.exit("size mismatch after download; re-run to resume")
    print(f"done: {DEST}")


if __name__ == "__main__":
    main()
