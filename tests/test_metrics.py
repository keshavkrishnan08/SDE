"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import crps_empirical, picp, pinaw, rmse, mae, skill_score
from src.evaluation.statistical_tests import diebold_mariano_test, holm_bonferroni_correction
from src.evaluation.calibration import probability_integral_transform, reliability_data


class TestCRPS:
    def test_perfect_forecast(self):
        y_true = np.array([5.0, 10.0, 15.0])
        # Perfect samples centered on true values
        y_samples = np.tile(y_true[:, None], (1, 100)) + np.random.randn(3, 100) * 0.001
        crps = crps_empirical(y_true, y_samples)
        assert all(c < 0.1 for c in crps)

    def test_crps_nonnegative(self):
        y_true = np.random.randn(50)
        y_samples = np.random.randn(50, 100)
        crps = crps_empirical(y_true, y_samples)
        assert (crps >= 0).all()


class TestPICP:
    def test_high_coverage(self):
        y_true = np.zeros(100)
        # Samples spread widely around 0
        y_samples = np.random.randn(100, 1000)
        coverage = picp(y_true, y_samples, alpha=0.90)
        assert coverage > 0.85  # Should cover well

    def test_narrow_coverage(self):
        y_true = np.ones(100) * 100
        # Samples far from true
        y_samples = np.random.randn(100, 100)
        coverage = picp(y_true, y_samples, alpha=0.90)
        assert coverage < 0.5  # Should miss most


class TestPointMetrics:
    def test_rmse_perfect(self):
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        assert rmse(y, y) == 0.0

    def test_mae_perfect(self):
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        assert mae(y, y) == 0.0

    def test_rmse_known(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 4.0])
        expected = np.sqrt((1 + 4) / 2)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-6


class TestSkillScore:
    def test_perfect_skill(self):
        assert skill_score(0.0, 10.0) == 1.0

    def test_no_skill(self):
        assert skill_score(10.0, 10.0) == 0.0

    def test_worse_than_reference(self):
        assert skill_score(15.0, 10.0) < 0


class TestDieboldMariano:
    def test_identical_errors(self):
        errors = np.random.randn(100) ** 2
        result = diebold_mariano_test(errors, errors)
        assert abs(result["statistic"]) < 0.1
        assert result["p_value"] > 0.5

    def test_significantly_different(self):
        errors_good = np.random.randn(100) ** 2 * 0.1
        errors_bad = np.random.randn(100) ** 2 * 10
        result = diebold_mariano_test(errors_good, errors_bad)
        assert result["p_value"] < 0.05


class TestHolmBonferroni:
    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        results = holm_bonferroni_correction(p_values)
        assert all(r["significant"] for r in results)

    def test_none_significant(self):
        p_values = [0.5, 0.6, 0.7]
        results = holm_bonferroni_correction(p_values)
        assert not any(r["significant"] for r in results)


class TestCalibration:
    def test_pit_range(self):
        y_true = np.random.randn(100)
        y_samples = np.random.randn(100, 200)
        pit = probability_integral_transform(y_true, y_samples)
        assert (pit >= 0).all()
        assert (pit <= 1).all()

    def test_reliability_shape(self):
        y_true = np.random.randn(100)
        y_samples = np.random.randn(100, 200)
        rel = reliability_data(y_true, y_samples)
        assert len(rel["nominal"]) == len(rel["observed"])
