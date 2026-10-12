from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import urllib.request
import numpy as np

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ARCHIVE = "uci-har.zip"
def _safe_extract(archive: Path, destination: Path):
    root = destination.resolve()
    with ZipFile(archive) as zipped:
        for item in zipped.infolist():
            parts = PurePosixPath(item.filename).parts
            if not parts or PurePosixPath(item.filename).is_absolute() or ".." in parts: raise ValueError(f"Unsafe archive member: {item.filename}")
            target = (root / Path(*parts)).resolve()
            if target != root and root not in target.parents: raise ValueError(f"Archive member escapes data directory: {item.filename}")
            if item.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(zipped.read(item))

def ensure_data(root: str) -> Path:
    root = Path(root); folder = root / "UCI HAR Dataset"
    if folder.is_dir(): return folder
    root.mkdir(parents=True, exist_ok=True); archive = root / ARCHIVE
    if not archive.exists():
        temporary = root / (ARCHIVE + ".partial")
        try: urllib.request.urlretrieve(URL, temporary); temporary.replace(archive)
        finally: temporary.unlink(missing_ok=True)
    _safe_extract(archive, root)
    if not folder.is_dir(): raise RuntimeError("UCI archive did not contain UCI HAR Dataset/")
    return folder
def load_split(root: Path, split: str):
    folder = root / split
    x = np.loadtxt(folder / f"X_{split}.txt", dtype=np.float32)
    y = np.loadtxt(folder / f"y_{split}.txt", dtype=np.int64) - 1
    subjects = np.loadtxt(folder / f"subject_{split}.txt", dtype=np.int64)
    if not (len(x) == len(y) == len(subjects)): raise ValueError(f"Inconsistent {split} split row counts")
    return x, y, subjects
