import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "ml-latest-small"
ARTIFACTS = ROOT / "artifacts"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_SHA256 = "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
METHODS = ("content", "item_cf", "popular", "genre")
POSITIVE_RATING = 4.0
NEIGHBORS = 40
SHRINKAGE = 10.0
MIN_COMMON = 2
PRIOR_STRENGTH = 20.0
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "4")
