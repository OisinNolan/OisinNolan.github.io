"""Upload the generated Etymap data (/etymap/data/) to the website-files Hugging Face Space.

    _etymap/.venv/bin/hf auth login              # once; needs a token with write access
    _etymap/.venv/bin/python _etymap/hf_upload.py

The files end up at https://oisinn-website-files.static.hf.space/etymap/data/…, which is
where /etymap/app.js reads them from when the page isn't running on localhost.
Stale files from earlier builds are removed in the same commit.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "oisinn/website_files"
REPO_TYPE = "space"
LOCAL = Path(__file__).resolve().parent.parent / "etymap" / "data"
REMOTE = "etymap/data"


def main():
    if not (LOCAL / "meta.json").exists():
        sys.exit(f"No data at {LOCAL}; run the build first (see _etymap/README.md).")
    n = sum(1 for _ in LOCAL.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in LOCAL.rglob("*") if f.is_file())
    print(f"Uploading {n:,} files ({size / 1e6:.0f} MB) to {REPO_TYPE} {REPO_ID}:{REMOTE} …")
    info = HfApi().upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=LOCAL,
        path_in_repo=REMOTE,
        delete_patterns=["**"],          # within REMOTE only: drop files no longer in the build
        commit_message="Update etymap data",
    )
    print("Done:", info)


if __name__ == "__main__":
    main()
