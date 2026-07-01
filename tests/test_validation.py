"""Offline tests for the Checkpoint-2 validation harness, metrics, and error analysis.

These run fully offline: metrics are checked against hand-computed values, and the
harness/error-analysis are exercised over the committed ground-truth cases + bundled
signal fixtures (no network).

Run with:  python -m pytest tests/  (or: python tests/test_validation.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from marketfit.groundtruth import GroundTruthLoader
from marketfit.validation import (
    ValidationHarness,
    analyze_errors,
    bucket_metrics,
    load_signal_fixtures,
    score_metrics,
    spearman,
)


# --- metrics --------------------------------------------------------------

def test_score_metrics_hand_computed():
    pred = [5, 4, 4, 3, 2]
    actual = [5, 4, 2, 3, 1]
    # errors = [0, 0, +2, 0, +1]
    m = score_metrics(pred, actual)
    assert m.n == 5
    assert m.mae == 0.6                       # (0+0+2+0+1)/5
    assert m.rmse == 1.0                      # sqrt((0+0+4+0+1)/5)
    assert m.exact_accuracy == 0.6            # 3/5 exact
    assert m.within_one_accuracy == 0.8       # 4/5 within +/-1
    assert m.bias == 0.6                      # model over-predicts on average
    assert m.rank_correlation > 0.0
    print("score_metrics OK ->", m)


def test_bucket_metrics_confusion():
    pred = [True, True, False, False]
    actual = [True, False, True, False]
    m = bucket_metrics(pred, actual)
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 1)
    assert m.accuracy == 0.5
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.f1 == 0.5
    print("bucket_metrics OK ->", m)


def test_spearman_monotonic_and_edge_cases():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([5, 5, 5], [1, 2, 3]) == 0.0   # constant vector -> undefined -> 0
    assert spearman([1], [1]) == 0.0               # n<2 -> 0


def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_metrics([1, 2], [1])
    with pytest.raises(ValueError):
        bucket_metrics([True], [True, False])


# --- harness --------------------------------------------------------------

def _cases_and_signals():
    return GroundTruthLoader().load(), load_signal_fixtures()


def test_harness_run_report_shape():
    cases, signals = _cases_and_signals()
    report = ValidationHarness().run(cases, signals)
    # Every fixture-backed case is scored; none skipped (fixtures cover all 7).
    assert report.score.n == len(cases)
    assert report.skipped == ()
    frame = report.to_frame()
    assert len(frame) == len(cases)
    assert list(frame.columns)[:4] == [
        "case_id", "country_iso3", "outcome_label", "actual_score",
    ]
    # Metrics are in-range and the per-case bucket flags are internally consistent.
    assert 0.0 <= report.bucket.accuracy <= 1.0
    for c in report.cases:
        assert (c.predicted_score - c.actual_score) == c.score_error
        assert c.bucket_correct == (c.predicted_success == c.actual_success)
    assert isinstance(report.summary(), str)
    print("harness OK ->", report.summary().splitlines()[0])


def test_harness_skips_cases_without_signals():
    cases, signals = _cases_and_signals()
    signals.pop("AUS")  # drop one country's fixtures
    report = ValidationHarness().run(cases, signals)
    assert "SBUX-AUS-2000" in report.skipped
    assert report.score.n == len(cases) - 1


def test_calibration_and_leave_one_out_run():
    cases, signals = _cases_and_signals()
    calibrated = ValidationHarness().run(cases, signals, calibrate=True)
    assert calibrated.calibrated_threshold is not None
    # In-sample calibration should not do worse than the uncalibrated bucket accuracy.
    baseline = ValidationHarness().run(cases, signals)
    assert calibrated.bucket.accuracy >= baseline.bucket.accuracy

    loo_bucket, loo_frame = ValidationHarness().leave_one_out(cases, signals)
    assert len(loo_frame) == len(cases)
    assert loo_bucket.n == len(cases)
    assert 0.0 <= loo_bucket.accuracy <= 1.0


# --- error analysis -------------------------------------------------------

def test_error_analysis_flags_exactly_the_errors():
    cases, signals = _cases_and_signals()
    report = ValidationHarness().run(cases, signals)
    analysis = analyze_errors(report, magnitude_threshold=2)

    flagged = {e.case_id for e in analysis.errors}
    for c in report.cases:
        is_error = (not c.bucket_correct) or abs(c.score_error) >= 2
        assert (c.case_id in flagged) == is_error

    # Aggregate counts line up with the per-case flags.
    assert analysis.over_predictions + analysis.under_predictions == len(analysis.errors)
    assert analysis.n_bucket_errors == sum(not c.bucket_correct for c in report.cases)
    assert isinstance(analysis.summary(), str)
    if analysis.errors:
        assert analysis.worst_case is not None
    print("error_analysis OK ->", analysis.summary().splitlines()[0])


def test_error_analysis_clean_report_has_no_errors():
    # A report whose predictions all match should yield an empty analysis.
    cases, signals = _cases_and_signals()
    report = ValidationHarness().run(cases, signals)
    # Force a "perfect" view by analyzing with an impossibly high magnitude threshold
    # and only bucket-correct cases: build a filtered report-like check instead.
    perfect = analyze_errors(
        ValidationReport_all_correct(report), magnitude_threshold=2
    )
    assert perfect.errors == []
    assert "No prediction errors" in perfect.summary()


def ValidationReport_all_correct(report):
    """Helper: a copy of ``report`` with predictions overwritten to match truth."""
    from dataclasses import replace
    fixed = [
        replace(
            c,
            predicted_score=c.actual_score,
            predicted_success=c.actual_success,
        )
        for c in report.cases
    ]
    return replace(report, cases=fixed)


if __name__ == "__main__":
    test_score_metrics_hand_computed()
    test_bucket_metrics_confusion()
    test_spearman_monotonic_and_edge_cases()
    test_metrics_length_mismatch_raises()
    test_harness_run_report_shape()
    test_harness_skips_cases_without_signals()
    test_calibration_and_leave_one_out_run()
    test_error_analysis_flags_exactly_the_errors()
    test_error_analysis_clean_report_has_no_errors()
    print("\nAll validation tests passed.")
