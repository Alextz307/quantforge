"""Validate UI-submitted configs + list/read YAMLs under ``config_root``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from src.core.config import (
    ExperimentConfig,
    StudySpec,
    UniverseProfile,
)
from src.core.hpo_config import HPOConfig
from webapp.backend.app.schemas.configs import (
    ConfigDetail,
    ConfigEntry,
    ConfigKind,
    ValidateResponse,
    ValidationErrorItem,
)
from webapp.backend.app.services.strategy_service import describe_strategy

__all__ = [
    "ConfigNotFoundError",
    "get_study_spec_schema",
    "list_configs",
    "read_config",
    "validate",
]

# Strategy YAMLs are loose dict bodies consumed by component ctors at
# runtime; there's no Pydantic class to validate them against without
# re-implementing the registry's coercion. ``None`` here means
# "validation degrades to YAML-parse only".
_KIND_TO_MODEL: dict[ConfigKind, type[BaseModel] | None] = {
    ConfigKind.EXPERIMENT: ExperimentConfig,
    ConfigKind.UNIVERSE: UniverseProfile,
    ConfigKind.STUDY: StudySpec,
    ConfigKind.HPO: HPOConfig,
    ConfigKind.STRATEGY: None,
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
}


class ConfigNotFoundError(FileNotFoundError):
    """Raised when ``config_root/<kind>/<name>.yaml`` is missing."""


def get_study_spec_schema() -> dict[str, object]:
    """Return the JSON Schema for :class:`StudySpec`.

    Surfaced via ``GET /api/configs/study_spec/schema`` and fed to
    ``monaco-yaml`` on the frontend so the editor gets autocomplete,
    hover docs, and inline type / required-field markers from the same
    Pydantic source of truth the CLI validates against.
    """
    return StudySpec.model_json_schema()


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

    For loose-bodied kinds (strategy) the response is always
    ``valid=True`` — there is no Pydantic counterpart and the deeper
    coercion happens when the CLI actually instantiates the component.

    For ``ConfigKind.EXPERIMENT``, after Pydantic accepts the payload we
    also walk the strategy ctor's required params (via the same
    ``describe_strategy`` introspection that drives the form schema) and
    surface any missing ones — Pydantic treats ``strategy.params`` as a
    generic mapping, so a missing ctor kwarg would otherwise only blow
    up at strategy-build time as a TypeError after the subprocess spawn.
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
    if kind is ConfigKind.EXPERIMENT:
        sig_errors = _strategy_param_completeness_errors(payload)
        if sig_errors:
            return ValidateResponse(valid=False, errors=sig_errors)
    return ValidateResponse(valid=True, errors=[])


def _strategy_param_completeness_errors(
    payload: dict[str, object],
) -> list[ValidationErrorItem]:
    """Missing-required-param errors for the strategy referenced by ``payload``.

    Returns ``[]`` for unknown strategy names (already caught by
    Pydantic) and for strategies whose ctor requires nothing the user
    hasn't supplied.
    """
    strategy = payload.get("strategy")
    if not isinstance(strategy, dict):
        return []
    name = strategy.get("name")
    if not isinstance(name, str):
        return []
    try:
        schema = describe_strategy(name)
    except KeyError:
        return []
    params = strategy.get("params") or {}
    if not isinstance(params, dict):
        return []
    errors: list[ValidationErrorItem] = []
    for param in schema.params:
        if not param.required:
            continue
        present = param.name in params
        value_is_null = present and params[param.name] is None
        if not present or (value_is_null and not param.nullable):
            errors.append(
                ValidationErrorItem(
                    loc=["strategy", "params", param.name],
                    msg="field required",
                    type="missing",
                )
            )
    return errors
