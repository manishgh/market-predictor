"""Resolve explicitly relocated data without changing evidence path identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.path_integrity import (
    resolve_existing_file_inside,
    resolve_write_path_inside,
    verify_no_reparse_ancestry,
)

_WINDOWS_DEVICES = frozenset({
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    *(f"{prefix}{suffix}" for prefix in ("COM", "LPT") for suffix in "123456789¹²³"),
})
_DATA = Path("data")
_RELOCATABLE_NAMESPACES = frozenset(Path(name) for name in (
    "raw", "features", "research", "canonical", "external", "labels", "cache",
))
_IMPLEMENTATION_SNAPSHOTS = Path("research/implementation_snapshots")


def _path(raw: str | Path) -> Path:
    name = str(raw)
    windows = PureWindowsPath(name)
    path = Path(name.replace("\\", "/"))
    components = windows.parts[1:] if windows.anchor else windows.parts
    if (
        not path.parts
        or name.startswith(("//", "\\"))
        or (windows.drive and not (windows.root and path.is_absolute()))
        or (windows.root and not path.is_absolute())
        or any(
            part == ".."
            or part.endswith((".", " "))
            or any(ord(character) < 32 or character in '<>:"|?*' for character in part)
            or part.split(".", 1)[0].rstrip().upper() in _WINDOWS_DEVICES
            for part in components
        )
    ):
        raise DataReadinessError(f"unsafe evidence path: {raw}")
    return path


def _directory(raw: Path, *, label: str) -> Path:
    path = verify_no_reparse_ancestry(_path(raw), label=label)
    if not path.is_dir():
        raise DataReadinessError(f"{label} is not an existing directory: {path}")
    return path


def _relocatable(path: Path) -> bool:
    return (
        bool(path.parts)
        and Path(path.parts[0]) in _RELOCATABLE_NAMESPACES
        and not path.is_relative_to(_IMPLEMENTATION_SNAPSHOTS)
    )


@dataclass(frozen=True, slots=True, init=False)
class EvidencePaths:
    """Map the dataset namespaces to a selected data directory.

    The selected directory mirrors the raw, features, research, canonical,
    external, labels and cache directories under project/data. Research
    implementation snapshots and all other namespaces stay project-local.
    """

    project_root: Path
    data_root: Path

    def __init__(self, project_root: Path, data_root: Path | None = None) -> None:
        project = _directory(project_root, label="project root")
        logical_data = project / _DATA
        storage = logical_data if data_root is None else _directory(data_root, label="data root")
        if storage != logical_data and (storage.is_relative_to(project) or project.is_relative_to(storage)):
            raise DataReadinessError("custom data root must be outside the project root and its ancestors")
        object.__setattr__(self, "project_root", project)
        object.__setattr__(self, "data_root", storage)

    def identity(self, raw: str | Path) -> Path:
        """Return the absolute logical project identity without opening a file."""
        path = self.project_root / _path(raw)
        logical_data = self.project_root / _DATA
        if self.data_root != logical_data and path.is_relative_to(self.data_root):
            relative = path.relative_to(self.data_root)
            if not _relocatable(relative):
                raise DataReadinessError(f"evidence path is outside the relocated dataset namespaces: {raw}")
            return logical_data / relative
        if path == self.project_root or not path.is_relative_to(self.project_root):
            raise DataReadinessError(f"evidence path is outside the project namespace: {raw}")
        return path

    def relative(self, raw: str | Path) -> Path:
        """Return the logical project-relative identity."""
        return self.identity(raw).relative_to(self.project_root)

    def _authority(self, raw: str | Path) -> tuple[Path, Path]:
        relative = self.relative(raw)
        if relative.is_relative_to(_DATA):
            data_relative = relative.relative_to(_DATA)
            if self.data_root != self.project_root / _DATA and _relocatable(data_relative):
                return self.data_root, data_relative
        return self.project_root, relative

    def read(self, raw: str | Path) -> Path:
        """Resolve an existing regular file under the selected authority."""
        root, relative = self._authority(raw)
        try:
            return resolve_existing_file_inside(root, str(relative), label="evidence")
        except OSError as exc:
            raise DataReadinessError(f"evidence authority is unavailable: {root}") from exc

    def child_identity(self, parent: str | Path, raw: str | Path) -> Path:
        """Return a logical identity strictly below the declared parent authority."""
        authority = self.identity(parent)
        path = self.identity(authority / _path(raw))
        if path == authority or not path.is_relative_to(authority):
            raise DataReadinessError(f"evidence child escapes its authority: {raw}")
        return path

    def read_child(self, parent: str | Path, raw: str | Path) -> Path:
        """Read a child under an explicit authority without following links."""
        return self.read(self.child_identity(parent, raw))

    def write(self, raw: str | Path) -> Path:
        """Resolve a prospective destination under the selected authority."""
        root, relative = self._authority(raw)
        return resolve_write_path_inside(root, relative, label="evidence destination")

    def directory(self, raw: str | Path) -> Path:
        """Resolve an existing directory under the selected authority."""
        root, relative = self._authority(raw)
        path = resolve_write_path_inside(root, relative, label="evidence directory")
        if not path.is_dir():
            raise DataReadinessError(f"evidence directory is missing: {path}")
        return path
