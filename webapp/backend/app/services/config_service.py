"""Validate UI-submitted configs + list/read YAMLs under ``config_root``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from src.core.config import (
    ExperimentConfig,
    StandaloneModelConfig,
    StudySpec,
    UniverseProfile,
)
from webapp.backend.app.schemas.configs import (
    ConfigDetail,
    ConfigEntry,
    ConfigKind,
    ValidateResponse,
    ValidationErrorItem,
)

# Strategy/HPO/regime YAMLs are loose dict bodies consumed by component
# ctors at runtime; there's no Pydantic class to validate them against
# without re-implementing the registry's coercion. ``None`` here means
# "validation degrades to YAML-parse only".
_KIND_TO_MODEL: dict[ConfigKind, type[BaseModel] | None] = {
    ConfigKind.EXPERIMENT: ExperimentConfig,
    ConfigKind.UNIVERSE: UniverseProfile,
    ConfigKind.STUDY: StudySpec,
    ConfigKind.MODEL: StandaloneModelConfig,
    ConfigKind.STRATEGY: None,
    ConfigKind.HPO: None,
    ConfigKind.REGIME: None,
}

# `config/` directory names are not 1:1 with kind values — some are
# plural, some not, and "experiment" has no top-level dir (experiment
# configs are deep-merged from strategy + universe at runtime). Listing
# under a missing dir returns [] rather than 404.
_KIND_TO_DIRNAME: dict[ConfigKind, str] = {
    ConfigKind.EXPERIMENT: "experiment",
    ConfigKind.UNIVERSE: "universes",
    ConfigKind.STRATEGY: "strategies",
    ConfigKind.HPO: "hpo",
    ConfigKind.STUDY: "study",
    ConfigKind.REGIME: "regimes",
    ConfigKind.MODEL: "models",
}


class ConfigNotFoundError(FileNotFoundError):
    """Raised when ``config_root/<kind>/<name>.yaml`` is missing."""


def _kind_dir(config_root: Path, kind: ConfigKind) -> Path:
    return config_root / _KIND_TO_DIRNAME[kind]


def list_configs(config_root: Path, kind: ConfigKind) -> list[ConfigEntry]:
    """List every ``*.yaml`` under ``config_root/<kind>/``, sorted by stem."""
    return [ConfigEntry(name=p.stem) for p in sorted(_kind_dir(config_root, kind).glob("*.yaml"))]


def read_config(config_root: Path, kind: ConfigKind, name: str) -> ConfigDetail:
    """Return the raw YAML text + best-effort parse for ``<name>.yaml``."""
    path = _kind_dir(config_root, kind) / f"{name}.yaml"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigNotFoundError(f"config not found: {path}") from exc
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        return ConfigDetail(name=name, raw=raw, parsed=None, parse_error=str(exc))
    if parsed is None:
        return ConfigDetail(name=name, raw=raw, parsed=None, parse_error="empty YAML")
    if not isinstance(parsed, dict):
        return ConfigDetail(
            name=name,
            raw=raw,
            parsed=None,
            parse_error=f"top-level YAML must be a mapping, got {type(parsed).__name__}",
        )
    return ConfigDetail(name=name, raw=raw, parsed=parsed, parse_error=None)


def validate(kind: ConfigKind, payload: dict[str, object]) -> ValidateResponse:
    """Validate ``payload`` against the Pydantic model bound to ``kind``.

    For loose-bodied kinds (strategy/hpo/regime) the response is always
    ``valid=True`` — there is no Pydantic counterpart and the deeper
    coercion happens when the CLI actually instantiates the component.
    """
    model_cls = _KIND_TO_MODEL[kind]
    if model_cls is None:
        return ValidateResponse(valid=True, errors=[])
    try:
        model_cls.model_validate(payload)
    except ValidationError as exc:
        return ValidateResponse(
            valid=False,
            errors=[
                ValidationErrorItem(
                    loc=[str(part) for part in err["loc"]],
                    msg=err["msg"],
                    type=err["type"],
                )
                for err in exc.errors()
            ],
        )
    return ValidateResponse(valid=True, errors=[])
