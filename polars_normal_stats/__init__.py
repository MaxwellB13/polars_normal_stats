from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars as pl

try:
    __version__ = version("polars-normal-stats")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["normal_cdf", "normal_ppf", "normal_pdf"]

# Find the compiled library
LIB_PATH = Path(__file__).parent


def _validate_scalar_params(mean: float, std: float) -> tuple[float, float]:
    try:
        return float(mean), float(std)
    except (TypeError, ValueError):
        # Handle cases where expressions or None are passed to maintain compatibility
        # with previous behavior that expected these to fail in Rust with a specific message.
        raise pl.exceptions.ComputeError(
            "mean and std must be a scalar value, not a column or null."
        ) from None


def normal_cdf(
    x: pl.Expr, mean: float = 0.0, std: float = 1.0
) -> pl.Expr:
    """
    Calculate the cumulative distribution function of the normal distribution.

    Parameters
    ----------
    x : pl.Expr
        The values at which to evaluate the CDF
    mean : float, default 0.0
        The mean of the normal distribution
    std : float, default 1.0
        The standard deviation of the normal distribution

    Returns
    -------
    pl.Expr
        The CDF values
    """
    mean_val, std_val = _validate_scalar_params(mean, std)

    return pl.plugins.register_plugin_function(
        plugin_path=LIB_PATH,
        function_name="normal_cdf",
        args=[x.cast(pl.Float64)],
        kwargs={"mean": mean_val, "std": std_val},
        is_elementwise=True,
    )


def normal_ppf(
    p: pl.Expr, mean: float = 0.0, std: float = 1.0
) -> pl.Expr:
    """
    Calculate the percent point function (inverse CDF) of the normal distribution.

    Parameters
    ----------
    p : pl.Expr
        The probability values (must be between 0 and 1)
    mean : float, default 0.0
        The mean of the normal distribution
    std : float, default 1.0
        The standard deviation of the normal distribution

    Returns
    -------
    pl.Expr
        The PPF values
    """
    mean_val, std_val = _validate_scalar_params(mean, std)

    return pl.plugins.register_plugin_function(
        plugin_path=LIB_PATH,
        function_name="normal_ppf",
        args=[p.cast(pl.Float64)],
        kwargs={"mean": mean_val, "std": std_val},
        is_elementwise=True,
    )


def normal_pdf(
    x: pl.Expr, mean: float = 0.0, std: float = 1.0
) -> pl.Expr:
    """
    Calculate the probability density function of the normal distribution.

    Parameters
    ----------
    x : pl.Expr
        The values at which to evaluate the PDF
    mean : float, default 0.0
        The mean of the normal distribution
    std : float, default 1.0
        The standard deviation of the normal distribution

    Returns
    -------
    pl.Expr
        The PDF values
    """
    mean_val, std_val = _validate_scalar_params(mean, std)

    return pl.plugins.register_plugin_function(
        plugin_path=LIB_PATH,
        function_name="normal_pdf",
        args=[x.cast(pl.Float64)],
        kwargs={"mean": mean_val, "std": std_val},
        is_elementwise=True,
    )
