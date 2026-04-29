"""Unit tests for src/core/persistence.py helpers (model-persistence layout +
scaler round-trip). Pure JSON helpers live in ``src.core.json_io`` and are
covered by ``tests/unit/test_json_io.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.core.persistence import (
    ensure_model_dir,
    load_standard_scaler,
    save_standard_scaler,
)

# Fixture scale constants — kept small; the helpers are pure IO and don't need
# large inputs to exercise the round-trip.
N_SAMPLES = 20
N_FEATURES = 3
SCALER_MEAN_SEED = 42
ROUND_TRIP_ATOL = 0.0


class TestEnsureModelDir:
    def test_creates_missing_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "fresh"
        result = ensure_model_dir(target)
        assert target.is_dir()
        assert result == target

    def test_reuses_empty_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "empty"
        target.mkdir()
        # Must not raise; empty existing dir is a valid save target.
        ensure_model_dir(target)

    def test_raises_on_non_empty_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "populated"
        target.mkdir()
        (target / "junk.txt").write_text("junk")
        with pytest.raises(FileExistsError, match="non-empty"):
            ensure_model_dir(target)

    def test_raises_on_file(self, tmp_path: Path) -> None:
        target = tmp_path / "file"
        target.write_text("not a dir")
        with pytest.raises(NotADirectoryError):
            ensure_model_dir(target)


class TestStandardScalerRoundTrip:
    def test_fitted_scaler_round_trips(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(SCALER_MEAN_SEED)
        data = rng.normal(0.0, 1.0, size=(N_SAMPLES, N_FEATURES))
        scaler = StandardScaler()
        scaler.fit(data)

        path = tmp_path / "scaler.json"
        save_standard_scaler(scaler, path)
        loaded = load_standard_scaler(path)

        assert loaded.n_features_in_ == N_FEATURES
        np.testing.assert_array_equal(loaded.mean_, scaler.mean_)
        np.testing.assert_array_equal(loaded.scale_, scaler.scale_)
        np.testing.assert_array_equal(loaded.var_, scaler.var_)
        # The decisive round-trip check: transform output must match.
        np.testing.assert_allclose(
            loaded.transform(data),
            scaler.transform(data),
            atol=ROUND_TRIP_ATOL,
            rtol=0.0,
        )

    def test_unfitted_scaler_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="unfitted"):
            save_standard_scaler(StandardScaler(), tmp_path / "never.json")

    def test_feature_names_roundtrip_silences_transform_warning(self, tmp_path: Path) -> None:
        """A scaler fit on a DataFrame carries ``feature_names_in_``; the
        save/load round-trip must preserve it so post-load ``.transform()``
        on a named DataFrame doesn't trip sklearn's "fit without feature
        names" warning.
        """
        rng = np.random.default_rng(SCALER_MEAN_SEED)
        frame = pd.DataFrame(
            rng.normal(0.0, 1.0, size=(N_SAMPLES, N_FEATURES)),
            columns=[f"feat_{i}" for i in range(N_FEATURES)],
        )
        scaler = StandardScaler().fit(frame)

        path = tmp_path / "scaler_with_names.json"
        save_standard_scaler(scaler, path)
        loaded = load_standard_scaler(path)

        np.testing.assert_array_equal(loaded.feature_names_in_, scaler.feature_names_in_)
        # Decisive check: calling transform() on a named DataFrame must not
        # emit any warning — the restored feature_names_in_ satisfies sklearn.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded.transform(frame)

    def test_round_trip_with_nan_input_handles_per_feature_n_samples_seen(
        self, tmp_path: Path
    ) -> None:
        """Sklearn's ``n_samples_seen_`` is a 1-D ndarray when the fit input
        contains NaN values (per-feature non-NaN counts), not a scalar int.
        The production feature pipeline produces leading warmup NaNs, so this
        is the live path — the round-trip must serialize the array shape.
        """
        rng = np.random.default_rng(SCALER_MEAN_SEED)
        data = rng.normal(0.0, 1.0, size=(N_SAMPLES, N_FEATURES))
        # Punch a different number of NaNs into each column so sklearn's
        # per-feature counts diverge — defeats any "all features saw the
        # same N samples" shortcut.
        data[0, 0] = np.nan
        data[0:2, 1] = np.nan
        scaler = StandardScaler().fit(data)
        # Precondition for the regression: this fit must produce the array
        # form, otherwise the test isn't exercising the bug shape.
        assert isinstance(scaler.n_samples_seen_, np.ndarray)
        assert scaler.n_samples_seen_.shape == (N_FEATURES,)

        path = tmp_path / "scaler_nan.json"
        save_standard_scaler(scaler, path)
        loaded = load_standard_scaler(path)

        np.testing.assert_array_equal(loaded.n_samples_seen_, scaler.n_samples_seen_)
        np.testing.assert_allclose(
            loaded.transform(data),
            scaler.transform(data),
            atol=ROUND_TRIP_ATOL,
            rtol=0.0,
            equal_nan=True,
        )

    def test_fit_on_ndarray_does_not_persist_feature_names(self, tmp_path: Path) -> None:
        """A scaler fit on a bare ndarray has no ``feature_names_in_``; the
        persisted JSON must omit the key (not write a null) and the loaded
        scaler must likewise lack the attribute.
        """
        rng = np.random.default_rng(SCALER_MEAN_SEED)
        scaler = StandardScaler().fit(rng.normal(0.0, 1.0, size=(N_SAMPLES, N_FEATURES)))

        path = tmp_path / "scaler_no_names.json"
        save_standard_scaler(scaler, path)
        loaded = load_standard_scaler(path)

        assert not hasattr(loaded, "feature_names_in_")
