import pytest

import polars as pl
import math
from polars_normal_stats import normal_cdf, normal_ppf, normal_pdf


def test_normal_cdf():
    df = pl.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0, None]})
    result = df.with_columns(cdf=normal_cdf(pl.col("x")))

    # Standard normal CDF at 0 should be 0.5
    assert abs(result.filter(pl.col("x") == 0.0)["cdf"][0] - 0.5) < 1e-10
    assert result["cdf"][5] is None


def test_normal_cdf_custom_params():
    df = pl.DataFrame({"x": [10.0, 11.0, 12.0]})
    # Mean 10, std 2. CDF(10) should be 0.5
    result = df.with_columns(cdf=normal_cdf(pl.col("x"), mean=10.0, std=2.0))
    assert abs(result.filter(pl.col("x") == 10.0)["cdf"][0] - 0.5) < 1e-10


def test_normal_ppf():
    df = pl.DataFrame({"p": [0.025, 0.5, 0.975, -0.1, 1.1, None]})
    result = df.with_columns(ppf=normal_ppf(pl.col("p")))

    # Standard normal PPF at 0.5 should be 0
    assert abs(result.filter(pl.col("p") == 0.5)["ppf"][0]) < 1e-10
    assert result["ppf"][3] is None
    assert result["ppf"][4] is None
    assert result["ppf"][5] is None


def test_lazy_execution():
    df = pl.DataFrame({"x": [float(i) for i in range(-10, 11)]})
    result = df.lazy().with_columns(cdf=normal_cdf(pl.col("x"))).collect()

    assert len(result) == 21


def test_normal_pdf():
    df = pl.DataFrame({"x": [0.0]})
    result = df.with_columns(pdf=normal_pdf(pl.col("x")))

    # Standard normal PDF at 0 should be 1/sqrt(2*pi)
    expected = 1.0 / math.sqrt(2.0 * math.pi)
    assert abs(result["pdf"][0] - expected) < 1e-10


def test_normal_ppf_boundaries():
    df = pl.DataFrame({"p": [0.0, 1.0]})
    result = df.with_columns(ppf=normal_ppf(pl.col("p")))
    
    # Standard normal PPF at 0 should be -inf, at 1 should be inf
    assert math.isinf(result["ppf"][0]) and result["ppf"][0] < 0
    assert math.isinf(result["ppf"][1]) and result["ppf"][1] > 0


def test_int_input():
    df = pl.DataFrame({"x": [0, 1, 2]})
    result = df.with_columns(cdf=normal_cdf(pl.col("x")))
    assert result["cdf"].dtype == pl.Float64
    assert abs(result["cdf"][0] - 0.5) < 1e-10


def test_normal_ppf_rejects_zero_std():
    df = pl.DataFrame({"p": [0.5, 0.9]})
    with pytest.raises(pl.exceptions.ComputeError):
        df.with_columns(ppf=normal_ppf(pl.col("p"), std=0.0))


def test_rejects_negative_std():
    df = pl.DataFrame({"x": [0.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="Standard deviation must be positive"):
        df.with_columns(cdf=normal_cdf(pl.col("x"), std=-1.0))


def test_rejects_column_parameters():
    df = pl.DataFrame({"x": [0.0, 1.0], "mean": [0.0, 1.0]})
    with pytest.raises(pl.exceptions.ComputeError, match="must be a scalar value, not a column"):
        df.with_columns(cdf=normal_cdf(pl.col("x"), mean=pl.col("mean")))


def test_rejects_null_parameters():
    df = pl.DataFrame({"x": [0.0]})
    # When mean is None, pl.lit(None) produces a Null type series.
    # The plugin expects Float64 and fails if it gets Null.
    with pytest.raises(pl.exceptions.ComputeError):
        df.with_columns(cdf=normal_cdf(pl.col("x"), mean=None))

