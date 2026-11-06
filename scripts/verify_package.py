"""Verify the SHA-256 manifest of this source package."""

import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    failures = []
    checked = 0
    for line in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        actual = digest(path) if path.is_file() else "MISSING"
        checked += 1
        if actual != expected:
            failures.append((relative, expected, actual))
    if failures:
        for item in failures:
            print("FAIL", *item)
        raise SystemExit(1)
    print(f"OK: {checked} files match MANIFEST.sha256")


if __name__ == "__main__":
    main()
