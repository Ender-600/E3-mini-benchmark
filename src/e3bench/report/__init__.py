"""Reporting and visualization utilities for E³ Mini-Benchmark."""

from .aggregate import aggregate_results
from .signif_test import run_significance_tests
from .plots import generate_plots

__all__ = [
    "aggregate_results",
    "run_significance_tests",
    "generate_plots"
]
