from pathlib import Path
from typing import Union
import polars as pl

__version__ = "0.1.0"
__all__ = ["normal_cdf", "normal_ppf", "normal_pdf"]

# Find the compiled library
LIB_PATH = Path(__file__).parent


def normal_cdf(
    x: pl.Expr, mean: Union[pl.Expr, float] = 0.0, std: Union[pl.Expr, float] = 1.0
) -> pl.Expr:
    """
    Calculate the cumulative distribution function of the normal distribution.

    Parameters
    ----------
    x : pl.Expr
        The values at which to evaluate the CDF
    mean : pl.Expr or float, default 0.0
        The mean of the normal distribution
    std : pl.Expr or float, default 1.0
        The standard deviation of the normal distribution

    Returns
    -------
    pl.Expr
        The CDF values
    """
    if not isinstance(mean, pl.Expr):
        mean = pl.lit(mean)
    if not isinstance(std, pl.Expr):
        std = pl.lit(std)

    return pl.plugins.register_plugin_function(
        plugin_path=LIB_PATH,
        function_name="normal_cdf",
        args=[x.cast(pl.Float64), mean, std],
        is_elementwise=True,
    )


def normal_ppf(
    p: pl.Expr, mean: Union[pl.Expr, float] = 0.0, std: Union[pl.Expr, float] = 1.0
) -> pl.Expr:
    """
    Calculate the percent point function (inverse CDF) of the normal distribution.

    Parameters
    ----------
    p : pl.Expr
        The probability values (must be between 0 and 1)
    mean : pl.Expr or float, default 0.0
        The mean of the normal distribution
    std : pl.Expr or float, default 1.0
        The standard deviation of the normal distribution

    Returns
    -------
    pl.Expr
        The PPF values
    """
    if not isinstance(mean, pl.Expr):
        mean = pl.lit(mean)
    if not isinstance(std, pl.Expr):
        std = pl.lit(std)

    return pl.plugins.register_plugin_function(
        plugin_path=LIB_PATH,
        function_name="normal_ppf",
        args=[p.cast(pl.Float64), mean, std],
        is_elementwise=True,
    )


def normal_pdf(
    x: pl.Expr, mean: Union[pl.Expr, float] = 0.0, std: Union[pl.Expr, float] = 1.0
) -> pl.Expr:
    """
    Calculate the probability density function of the normal distribution.

    Parameters
    ----------
    x : pl.Expr
        The values at which to evaluate the PDF
    mean : pl.Expr or float, default 0.0
        The mean of the normal distribution
    std : pl.Expr or float, default 1.0
        The standard deviation of the normal distribution

    Returns
    -------
    pl.Expr
        The PDF values
    """
    if not isinstance(mean, pl.Expr):
        mean = pl.lit(mean)
    if not isinstance(std, pl.Expr):
        std = pl.lit(std)

    return pl.plugins.register_plugin_function(
        plugin_path=LIB_PATH,
        function_name="normal_pdf",
        args=[x.cast(pl.Float64), mean, std],
        is_elementwise=True,
    )
