from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

ML_V3_SCHEMA_VERSION = "ml_v3.v1"
_SCHEMA_TOKEN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*$")


class FrozenContract(BaseModel):
    """Strict immutable base for persisted V3 contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SchemaIdentity(FrozenContract):
    name: str = Field(min_length=1, max_length=80)
    version: str = Field(default=ML_V3_SCHEMA_VERSION, min_length=1, max_length=80)

    @field_validator("name", "version")
    @classmethod
    def validate_token(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not _SCHEMA_TOKEN.fullmatch(normalized):
            raise ValueError("schema tokens must use lowercase letters, digits, underscores, and dots")
        return normalized
