"""
Torch device selection: CUDA > MPS > CPU, with optional explicit override.
"""

from __future__ import annotations

from functools import cache

import torch

from src.core.logging import get_logger
from src.core.types import Device

logger = get_logger(__name__)

_ALLOWED_PREFERENCES = tuple(d.value for d in Device)
_XGBOOST_ALLOWED = (Device.AUTO.value, Device.CUDA.value, Device.CPU.value)


def _cuda_available() -> bool:
    """
    True iff torch was built with CUDA support AND a CUDA device is visible.
    """

    return torch.cuda.is_available()


def _mps_available() -> bool:
    """
    True iff torch was built with MPS support AND Apple Silicon hardware is present.
    """

    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _require_cuda() -> None:
    """
    Raise ``RuntimeError`` if CUDA is unavailable — shared by both selectors.
    """

    if not _cuda_available():
        raise RuntimeError(
            "device preference='cuda' but CUDA is not available; fix by setting "
            "device='auto' (or omitting it) on a CUDA-less host, or by installing "
            "CUDA-enabled builds of the relevant frameworks."
        )


@cache
def available_devices() -> tuple[Device, ...]:
    """
    Devices the host can actually drive (always includes ``AUTO`` and ``CPU``).

    Host capabilities don't change at runtime, so the result is cached.
    Tests that monkeypatch ``_cuda_available`` / ``_mps_available`` must
    call ``available_devices.cache_clear()`` before asserting.
    """

    devices: list[Device] = [Device.AUTO, Device.CPU]
    if _cuda_available():
        devices.append(Device.CUDA)
    if _mps_available():
        devices.append(Device.MPS)
    return tuple(devices)


def select_device(preference: Device | None = None) -> torch.device:
    """
    Return a ``torch.device`` following the CUDA > MPS > CPU priority order.

    Args:
        preference: ``None`` or ``Device.AUTO`` auto-picks the fastest available backend.
            ``Device.CUDA`` / ``Device.MPS`` / ``Device.CPU`` force that backend,
            raising if the requested one is unavailable.

    Raises:
        ValueError: if ``preference`` is an unrecognized value.
        RuntimeError: if an explicitly-named backend is unavailable on this host.
    """

    if preference is None or preference == Device.AUTO:
        if _cuda_available():
            device = torch.device(Device.CUDA)
        elif _mps_available():
            device = torch.device(Device.MPS)
        else:
            device = torch.device(Device.CPU)
        logger.info("auto-selected torch device: %s", device)
        return device

    if preference == Device.CUDA:
        _require_cuda()
        return torch.device(Device.CUDA)
    if preference == Device.MPS:
        if not _mps_available():
            raise RuntimeError(
                "device preference='mps' but MPS is not available; fix by setting "
                "device='auto' (or omitting it) on a non-Apple-Silicon host, or by "
                "upgrading to a torch build with MPS support."
            )
        return torch.device(Device.MPS)
    if preference == Device.CPU:
        return torch.device(Device.CPU)

    raise ValueError(
        f"device preference must be one of {_ALLOWED_PREFERENCES}, got "
        f"{preference!r}; fix by passing a Device enum value (or None for auto)."
    )


def select_xgboost_device(preference: Device | None = None) -> str:
    """
    Return the XGBoost device string (``'cuda'`` or ``'cpu'``).

    XGBoost's CUDA support is NVIDIA-only — it has no Apple MPS backend — so
    MPS is intentionally omitted from the priority order and rejected with a
    clear error if requested explicitly.

    Args:
        preference: ``None`` or ``Device.AUTO`` auto-picks CUDA > CPU.
            ``Device.CUDA`` / ``Device.CPU`` force that backend.

    Raises:
        ValueError: unrecognized value, including ``Device.MPS``.
        RuntimeError: ``Device.CUDA`` requested but CUDA is not available.
    """

    if preference is None or preference == Device.AUTO:
        device = Device.CUDA.value if _cuda_available() else Device.CPU.value
        logger.info("auto-selected XGBoost device: %s", device)
        return device
    if preference == Device.CUDA:
        _require_cuda()
        return Device.CUDA.value
    if preference == Device.CPU:
        return Device.CPU.value
    if preference == Device.MPS:
        raise ValueError(
            "XGBoost has no MPS backend; fix by passing Device.CUDA or Device.CPU "
            "(MPS is torch-only — XGBoost's GPU path is NVIDIA-only)."
        )
    raise ValueError(
        f"XGBoost device preference must be one of {_XGBOOST_ALLOWED}, got "
        f"{preference!r}; fix by passing Device.CUDA, Device.CPU, or None for auto."
    )
