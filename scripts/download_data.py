"""Download an original TSB-AD archive; benchmark CSV manifests are under data/."""
import argparse
from pathlib import Path
from urllib.request import urlopen
import shutil
import zipfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["U", "M"], required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / ("TSB-AD-" + args.track + ".zip")
    url = "https://www.thedatum.org/datasets/" + archive.name
    partial = archive.with_suffix(".zip.partial")
    with urlopen(url, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(archive)
    root = args.output_dir.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Unsafe archive member: " + item.filename)
        bundle.extractall(root)
    print(root)
