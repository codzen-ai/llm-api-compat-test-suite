from __future__ import annotations

import os
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

import yaml
from pydantic import BaseModel, model_validator

if TYPE_CHECKING:
    from pathlib import Path


class ModelConfig(BaseModel):
    name: str
    capabilities: list[str] = ["chat"]

    @model_validator(mode="before")
    @classmethod
    def accept_string(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data}
        return data


class ApiFormat(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ProviderConfig(BaseModel):
    name: str = "default"
    base_url: str
    api_key: str = ""
    api_format: ApiFormat
    auth_type: str | None = None  # "bearer" | "x-api-key" | "x-goog-api-key"
    verify_ssl: bool = True
    models: list[ModelConfig] = []

    @model_validator(mode="before")
    @classmethod
    def resolve_api_key(cls, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return data  # type: ignore[return-value]
        d = cast("dict[str, Any]", data)
        if "api_key" not in d and "api_key_env" in d:
            env_var: str = d["api_key_env"]
            api_key = os.environ.get(env_var, "")
            if not api_key:
                msg = f"Environment variable '{env_var}' is not set"
                raise ValueError(msg)
            d["api_key"] = api_key
        if "base_url" in d:
            d["base_url"] = str(d["base_url"]).rstrip("/")
        return d


class SuiteConfig(BaseModel):
    providers: list[ProviderConfig] = []

    @classmethod
    def from_yaml(cls, path: Path) -> SuiteConfig:
        with path.open() as f:  # noqa: PTH123
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_cli(
        cls,
        base_url: str,
        api_key: str,
        api_format: str,
        model: str | None = None,
    ) -> SuiteConfig:
        """Build config from CLI arguments for single-provider testing."""
        models: list[ModelConfig] = []
        if model:
            models = [
                ModelConfig(
                    name=model,
                    capabilities=[
                        "chat",
                        "streaming",
                        "tools",
                        "vision",
                        "embeddings",
                    ],
                )
            ]

        provider = ProviderConfig(
            name="cli",
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            api_format=ApiFormat(api_format),
            models=models,
        )
        return cls(providers=[provider])
