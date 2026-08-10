"""Provenance manifests (spec detail §0.2).

Every phase writes a manifest.json alongside its output containing input
artifact hashes, a config snapshot, the git SHA, a timestamp, and the seed.
"A phase that cannot reproduce its inputs from the manifest is not done."
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import paths

_HASH_CHUNK = 1 << 20
# Hashing a multi-GB parquet on every phase boundary is not worth the wall
# clock; beyond this size we hash the first and last chunk plus the byte
# length, which still detects any realistic accidental change.
_FULL_HASH_MAX_BYTES = 512 << 20


def git_sha(short: bool = False) -> str:
    cmd = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        out = subprocess.run(
            cmd, cwd=paths.ROOT, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=paths.ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.stdout.strip())


def hash_file(path: Path) -> str:
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as fh:
        if size <= _FULL_HASH_MAX_BYTES:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
                h.update(chunk)
            return f"sha256:{h.hexdigest()}"
        h.update(fh.read(_HASH_CHUNK))
        fh.seek(-_HASH_CHUNK, 2)
        h.update(fh.read(_HASH_CHUNK))
    return f"sha256-partial:{h.hexdigest()}:{size}"


def hash_path(path: Path) -> str:
    """Hash a file, or a directory as the hash of its sorted (name, hash) list."""
    path = Path(path)
    if path.is_file():
        return hash_file(path)
    if not path.is_dir():
        return "missing"
    h = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(child.relative_to(path)).encode())
        h.update(hash_file(child).encode())
    return f"sha256-tree:{h.hexdigest()}"


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_manifest(
    out_dir: Path,
    *,
    phase: str,
    config: Any,
    seed: int,
    inputs: Iterable[Path] = (),
    outputs: Iterable[Path] = (),
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write out_dir/manifest.json and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "phase": phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": _jsonable(config),
        "inputs": {str(p): hash_path(Path(p)) for p in inputs},
        "outputs": {str(p): hash_path(Path(p)) for p in outputs},
    }
    if extra:
        manifest["extra"] = _jsonable(extra)

    dest = out_dir / "manifest.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    return dest
