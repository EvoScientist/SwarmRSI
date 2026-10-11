from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Callable


class RuntimeArtifactCollector:
    """Archive declared, newly generated logs outside the executable workspace.

    No model edit gains extra permissions. Existing files, executable files,
    source files, and symlinks remain in the workspace for normal contract checks.
    """

    DATA_SUFFIXES = frozenset({".json", ".jsonl", ".csv", ".tsv", ".txt", ".log", ".tfevents"})

    def __init__(self, directories: tuple[str, ...], archive: Path,
                 write_allowed: Callable[[str], bool], protected: tuple[str, ...]) -> None:
        self.directories = directories
        self.archive = archive
        self.write_allowed = write_allowed
        self.protected = protected
        self.command = 0
        self.last_receipt: dict | None = None

    def inventory(self, root: Path) -> set[str]:
        paths: set[str] = set()
        for directory in self.directories:
            base = root / directory
            if base.is_symlink() or base.is_file():
                paths.add(directory)
            elif base.is_dir():
                paths.update(p.relative_to(root).as_posix() for p in base.rglob("*"))
        return paths

    def _eligible(self, root: Path, relative: str) -> bool:
        path = root / relative
        cursor = root
        for part in PurePosixPath(relative).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return False
        if not path.is_file() or path.suffix not in self.DATA_SUFFIXES:
            return False
        if path.stat().st_mode & 0o111 or self.write_allowed(relative):
            return False
        return not any(
            PurePosixPath(relative).match(protected)
            or relative == protected or relative.startswith(protected.rstrip("/") + "/")
            for protected in self.protected
        )

    def collect(self, root: Path, before: set[str]) -> dict:
        self.command += 1
        self.last_receipt = None
        records = []
        sources = []
        destination = self.archive / f"command_{self.command:04d}"
        for relative in sorted(self.inventory(root) - before):
            if not self._eligible(root, relative):
                continue
            source = root / relative
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise FileExistsError(target)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            size = source.stat().st_size
            # Copy and verify before removing a generated log from the source
            # manifest. The source remains available if archival fails.
            shutil.copy2(source, target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("runtime artifact copy hash mismatch")
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError("runtime artifact changed during archival")
            sources.append((source, digest))
            records.append({
                "workspace_path": relative, "archive_path": target.relative_to(self.archive).as_posix(),
                "sha256": digest, "bytes": size,
            })
        receipt = {"schema_version": "evolvcoder-runtime-artifacts-v1", "files": records}
        if records:
            raw = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
            receipt["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
            path = destination / "manifest.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            temporary.replace(path)
            # Commit the whole receipt before removing any workspace byte.
            # A later deletion failure still has a durable, hash-bound archive.
            self.last_receipt = receipt
            for source, digest in sources:
                if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                    raise ValueError("runtime artifact changed before removal")
                source.unlink()
        return receipt
