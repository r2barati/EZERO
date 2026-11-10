"""Long-sequence forecasting (LSF) evaluation datasets and the zero-shot protocol.

Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic (thesis Table 2 / B.3).

Splits follow the standard Time-Series-Library / Informer convention:

* ETTh*: 12 / 4 / 4 months (hourly), ETTm*: the same in 15-minute steps;
* Weather, Electricity, Traffic: 70% / 10% / 20%.

Scaling: a per-channel StandardScaler fitted on the **train split only** (``scaler='train'``).
The thesis (Sec. 3.6.1) normalised with statistics of the whole series and flags this as
a source of leakage; ``scaler='full'`` reproduces that legacy choice for comparison.

Test windows are every start position (stride 1) of the test border, ``drop_last=False``.
"""
from __future__ import annotations

import glob
import hashlib
import os
import subprocess
import urllib.request
from dataclasses import dataclass

import numpy as np

ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
KAGGLE_LSF_DATASET = "giochelavaipiatti/time-series-forecasts-popular-benchmark-datasets"

# name -> file, freq, split rule, sha256 of the canonical CSV, expected (rows, channels)
LSF_DATASETS = {
    "ETTh1": dict(file="ETTh1.csv", freq="H", split="ett_hour", shape=(17420, 7),
                  sha256="f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"),
    "ETTh2": dict(file="ETTh2.csv", freq="H", split="ett_hour", shape=(17420, 7),
                  sha256="a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"),
    "ETTm1": dict(file="ETTm1.csv", freq="15T", split="ett_minute", shape=(69680, 7),
                  sha256="6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"),
    "ETTm2": dict(file="ETTm2.csv", freq="15T", split="ett_minute", shape=(69680, 7),
                  sha256="db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd"),
    "Weather": dict(file="weather.csv", freq="10T", split="ratio", shape=(52696, 21),
                    sha256="34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63"),
    "Electricity": dict(file="electricity.csv", freq="H", split="ratio", shape=(26304, 321),
                        sha256="7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad"),
    "Traffic": dict(file="traffic.csv", freq="H", split="ratio", shape=(17544, 862),
                    sha256="cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"),
}
DEFAULT_DATASETS = list(LSF_DATASETS)
DEFAULT_HORIZONS = (96, 192, 336, 720)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_data_dirs() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    dirs = []
    if os.environ.get("EZERO_DATA_DIR"):
        dirs.append(os.environ["EZERO_DATA_DIR"])
    dirs += [os.path.join(root, "data", "raw"), "/kaggle/input"]
    return dirs


def find_file(filename: str, dirs: list[str] | None = None) -> str | None:
    for d in dirs or default_data_dirs():
        if not os.path.isdir(d):
            continue
        direct = os.path.join(d, filename)
        if os.path.exists(direct):
            return direct
        hits = [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                if os.path.basename(p).lower() == filename.lower()]
        if hits:
            return sorted(hits)[0]
    return None


def fetch(name: str, dest_dir: str | None = None) -> str:
    """Locate a dataset CSV, downloading it if necessary (ETT: GitHub; others: Kaggle API)."""
    spec = LSF_DATASETS[name]
    path = find_file(spec["file"])
    if path:
        return path
    dest_dir = dest_dir or default_data_dirs()[-2]
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, spec["file"])
    if name.startswith("ETT"):
        urllib.request.urlretrieve(ETT_URL.format(name=name), out)
        return out
    cmd = ["kaggle", "datasets", "download", KAGGLE_LSF_DATASET, "-f", spec["file"], "-p", dest_dir, "--unzip"]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:  # pragma: no cover - network dependent
        raise FileNotFoundError(
            f"{spec['file']} not found. Put it in data/raw/ or attach the Kaggle dataset "
            f"{KAGGLE_LSF_DATASET} (error: {exc})") from exc
    zipped = out + ".zip"
    if not os.path.exists(out) and os.path.exists(zipped):
        import zipfile
        with zipfile.ZipFile(zipped) as z:
            z.extractall(dest_dir)
    return out


@dataclass
class LSFData:
    name: str
    freq: str
    values: np.ndarray        # (T, C) float32 raw values
    borders: dict             # split -> (start, end) for a given context length
    mean: np.ndarray          # (C,)
    std: np.ndarray           # (C,)
    scaler: str
    context_length: int

    @property
    def channels(self) -> int:
        return self.values.shape[1]

    def scaled(self) -> np.ndarray:
        return ((self.values - self.mean) / self.std).astype(np.float32)

    def split_values(self, split: str, scaled: bool = True) -> np.ndarray:
        s, e = self.borders[split]
        v = self.scaled() if scaled else self.values
        return v[s:e]

    def n_windows(self, horizon: int, split: str = "test") -> int:
        s, e = self.borders[split]
        return max(0, (e - s) - self.context_length - horizon + 1)


def compute_borders(n: int, rule: str, L: int) -> dict:
    if rule == "ett_hour":
        m = 30 * 24
        b1 = [0, 12 * m - L, 12 * m + 4 * m - L]
        b2 = [12 * m, 12 * m + 4 * m, 12 * m + 8 * m]
    elif rule == "ett_minute":
        m = 30 * 24 * 4
        b1 = [0, 12 * m - L, 12 * m + 4 * m - L]
        b2 = [12 * m, 12 * m + 4 * m, 12 * m + 8 * m]
    elif rule == "ratio":
        n_train = int(n * 0.7)
        n_test = int(n * 0.2)
        n_val = n - n_train - n_test
        b1 = [0, n_train - L, n - n_test - L]
        b2 = [n_train, n_train + n_val, n]
    else:
        raise ValueError(rule)
    return {"train": (b1[0], b2[0]), "val": (b1[1], b2[1]), "test": (b1[2], b2[2])}


def load_lsf(name: str, context_length: int = 512, scaler: str = "train", path: str | None = None,
             verify: bool = True) -> LSFData:
    import pandas as pd

    spec = LSF_DATASETS[name]
    path = path or fetch(name)
    df = pd.read_csv(path)
    values = df.drop(columns=[df.columns[0]]).to_numpy(dtype=np.float32)
    if verify and tuple(values.shape) != tuple(spec["shape"]):
        raise ValueError(f"{name}: unexpected shape {values.shape}, expected {spec['shape']} ({path})")
    borders = compute_borders(len(values), spec["split"], context_length)
    if scaler == "train":
        s, e = borders["train"]
        ref = values[s:e]
    elif scaler == "full":
        ref = values
    else:
        raise ValueError("scaler must be 'train' or 'full'")
    mean = ref.mean(0)
    std = ref.std(0)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return LSFData(name, spec["freq"], values, borders, mean.astype(np.float32), std, scaler, context_length)


def window_starts(data: LSFData, protocol: str, horizon_max: int = 720, horizon_min: int = 96) -> np.ndarray:
    """Test-window start offsets (relative to the test border).

    ``full``  : every start valid for the shortest horizon (stride 1, standard LSF protocol).
    ``subN``  : N starts evenly spaced over the range valid for the longest horizon, shared by
                all horizons (used for expensive foundation-model baselines).
    """
    if protocol == "full":
        return np.arange(data.n_windows(horizon_min))
    if protocol.startswith("sub"):
        k = int(protocol[3:])
        n = data.n_windows(horizon_max)
        return np.unique(np.linspace(0, n - 1, num=min(k, n)).round().astype(int))
    if protocol.startswith("stride"):
        k = int(protocol[6:])
        return np.arange(0, data.n_windows(horizon_min), k)
    raise ValueError(f"unknown protocol {protocol!r}")
