import argparse
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from recsys.config import ROOT, DATA_URL, DATA_SHA256
from recsys.utils import sha256, save_json


def prepare(archive_path=None):
    archive = ROOT / "data/raw/ml-latest-small.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive_path:
        shutil.copyfile(archive_path, archive)
    elif not archive.exists():
        temporary = archive.with_suffix(".zip.part")
        with urllib.request.urlopen(DATA_URL, timeout=60) as response, open(
            temporary, "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        os.replace(temporary, archive)
    if sha256(archive) != DATA_SHA256:
        raise RuntimeError(
            "Dataset SHA256 changed. Verify the new upstream release before accepting it."
        )
    target = archive.parent.resolve()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            if not (target / info.filename).resolve().is_relative_to(target):
                raise RuntimeError("Unsafe path in dataset archive.")
        source.extractall(target)
    save_json(
        {
            "source": DATA_URL,
            "sha256": DATA_SHA256,
            "dataset": "MovieLens latest-small, release 2018-09-26",
            "citation": "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. DOI: 10.1145/2827872",
            "upstream_readme": "https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html",
            "note": "Raw data and fitted artifacts are excluded from Git. Research/education use; upstream license applies.",
        },
        ROOT / "data/DATA_SOURCE.json",
    )
    print("Verified MovieLens files prepared.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded archive.")
    args = parser.parse_args()
    prepare(args.archive)
