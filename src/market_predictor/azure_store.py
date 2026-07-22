from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from market_predictor.config import Settings


class AzureBlobStore:
    def __init__(self, settings: Settings) -> None:
        if not settings.has_azure_storage:
            raise ValueError("Azure storage is not configured. Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL.")
        from azure.storage.blob import BlobServiceClient

        if settings.azure_storage_connection_string:
            self.service = BlobServiceClient.from_connection_string(settings.azure_storage_connection_string)
        else:
            account_url = settings.azure_storage_account_url
            if not account_url:
                raise ValueError("AZURE_STORAGE_ACCOUNT_URL is required when a connection string is not configured.")
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as exc:
                raise ValueError(
                    "AZURE_STORAGE_ACCOUNT_URL requires azure-identity. Install azure-identity or use AZURE_STORAGE_CONNECTION_STRING."
                ) from exc
            self.service = BlobServiceClient(
                account_url=account_url,
                credential=DefaultAzureCredential(),
            )
        self.container = self.service.get_container_client(settings.azure_storage_container)
        self.prefix = settings.azure_prefix

    def blob_name(self, relative: str | Path) -> str:
        path = str(relative).replace("\\", "/").lstrip("/")
        return f"{self.prefix}/{path}" if self.prefix else path

    def upload_file(self, local_path: Path, blob_relative: str | Path, *, overwrite: bool = True) -> str:
        blob_name = self.blob_name(blob_relative)
        blob = self.container.get_blob_client(blob_name)
        with local_path.open("rb") as handle:
            blob.upload_blob(handle, overwrite=overwrite, metadata={"sha256": _file_sha256(local_path)})
        return blob_name

    def upload_bytes(
        self,
        data: bytes,
        blob_relative: str | Path,
        *,
        overwrite: bool = True,
    ) -> str:
        blob_name = self.blob_name(blob_relative)
        self.container.get_blob_client(blob_name).upload_blob(
            data,
            overwrite=overwrite,
            metadata={"sha256": hashlib.sha256(data).hexdigest()},
        )
        return blob_name

    def download_bytes(self, blob_relative: str | Path) -> bytes:
        blob = self.container.get_blob_client(self.blob_name(blob_relative))
        return bytes(blob.download_blob().readall())

    def blob_exists(self, blob_relative: str | Path) -> bool:
        return bool(self.container.get_blob_client(self.blob_name(blob_relative)).exists())

    def blob_sha256(self, blob_relative: str | Path) -> str | None:
        properties = self.container.get_blob_client(self.blob_name(blob_relative)).get_blob_properties()
        metadata = getattr(properties, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        value = metadata.get("sha256")
        return str(value) if value else None

    def download_file(self, blob_relative: str | Path, local_path: Path, *, overwrite: bool = True) -> Path:
        if local_path.exists() and not overwrite:
            return local_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = self.container.get_blob_client(self.blob_name(blob_relative))
        data = blob.download_blob().readall()
        local_path.write_bytes(data)
        return local_path

    def upload_tree(
        self,
        root: Path,
        *,
        blob_prefix: str | Path,
        patterns: Iterable[str] = ("*",),
        overwrite: bool = True,
    ) -> list[dict[str, str]]:
        uploaded: list[dict[str, str]] = []
        seen: set[Path] = set()
        for pattern in patterns:
            for path in root.rglob(pattern):
                if not path.is_file() or path in seen:
                    continue
                seen.add(path)
                relative = path.relative_to(root)
                blob_name = self.upload_file(path, Path(str(blob_prefix)) / relative, overwrite=overwrite)
                uploaded.append({"local_path": str(path), "blob": blob_name})
        return uploaded


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
