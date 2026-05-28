"""
Tests for torch device selection helper.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from src.core.device import available_devices, select_device, select_xgboost_device
from src.core.types import Device


class TestSelectDevice:
    def test_auto_prefers_cuda_when_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            device = select_device()
        assert device.type == "cuda"

    def test_auto_prefers_mps_over_cpu_when_cuda_absent(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.core.device._mps_available", return_value=True),
        ):
            device = select_device()
        assert device.type == "mps"

    def test_auto_falls_back_to_cpu_when_no_accelerator(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.core.device._mps_available", return_value=False),
        ):
            device = select_device()
        assert device.type == "cpu"

    def test_explicit_cpu_always_works(self) -> None:
        assert select_device(Device.CPU).type == "cpu"

    def test_explicit_auto_equivalent_to_none(self) -> None:
        assert select_device(Device.AUTO).type == select_device(None).type

    def test_explicit_cuda_raises_if_unavailable(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                select_device(Device.CUDA)

    def test_explicit_mps_raises_if_unavailable(self) -> None:
        with patch("src.core.device._mps_available", return_value=False):
            with pytest.raises(RuntimeError, match="MPS is not available"):
                select_device(Device.MPS)

    def test_returns_torch_device_instance(self) -> None:
        assert isinstance(select_device(Device.CPU), torch.device)


class TestSelectXGBoostDevice:
    def test_auto_prefers_cuda_when_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            assert select_xgboost_device() == "cuda"

    def test_auto_falls_back_to_cpu_without_cuda(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            assert select_xgboost_device() == "cpu"

    def test_auto_never_picks_mps(self) -> None:
        """
        Even on Apple Silicon with MPS available, XGBoost must fall back to CPU.
        """

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.core.device._mps_available", return_value=True),
        ):
            assert select_xgboost_device() == "cpu"

    def test_explicit_cpu_always_works(self) -> None:
        assert select_xgboost_device(Device.CPU) == "cpu"

    def test_explicit_cuda_raises_if_unavailable(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                select_xgboost_device(Device.CUDA)

    def test_mps_preference_rejected(self) -> None:
        with pytest.raises(ValueError, match="no MPS backend"):
            select_xgboost_device(Device.MPS)


class TestAvailableDevices:
    """
    ``available_devices`` is the predicate the webapp uses to prune the
    device dropdown. AUTO + CPU are always present; CUDA/MPS gated by host."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self) -> None:
        available_devices.cache_clear()

    def test_includes_auto_and_cpu_on_bare_host(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.core.device._mps_available", return_value=False),
        ):
            assert available_devices() == (Device.AUTO, Device.CPU)

    def test_includes_cuda_when_available(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("src.core.device._mps_available", return_value=False),
        ):
            assert Device.CUDA in available_devices()
            assert Device.MPS not in available_devices()

    def test_includes_mps_when_available(self) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("src.core.device._mps_available", return_value=True),
        ):
            assert Device.MPS in available_devices()
            assert Device.CUDA not in available_devices()
