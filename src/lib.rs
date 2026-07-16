use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use statrs::function::erf;

/// 1 / sqrt(2*pi), used to normalize the Gaussian PDF.
const FRAC_1_SQRT_2PI: f64 = 0.3989422804014326779399460599343818684758586311649346576659258296;

#[derive(Deserialize)]
struct NormalKwargs {
    mean: f64,
    std: f64,
}

/// Validates the distribution parameters so per-column invariants can be precomputed once instead of on every row.
fn validate_params(mean: f64, std: f64) -> PolarsResult<()> {
    if mean.is_nan() {
        return Err(PolarsError::ComputeError("Mean must not be NaN.".into()));
    }
    if std.is_nan() || std <= 0.0 {
        return Err(PolarsError::ComputeError(
            format!("Standard deviation must be positive. Got: {}", std).into(),
        ));
    }
    Ok(())
}

fn extract_ca<'a>(inputs: &'a [Series], kwargs: &NormalKwargs) -> PolarsResult<&'a Float64Chunked> {
    validate_params(kwargs.mean, kwargs.std)?;
    inputs[0].f64()
}

#[polars_expr(output_type=Float64)]
fn normal_cdf(inputs: &[Series], kwargs: NormalKwargs) -> PolarsResult<Series> {
    let ca = extract_ca(inputs, &kwargs)?;
    let mean = kwargs.mean;
    let inv_std_sqrt2 = 1.0 / (kwargs.std * std::f64::consts::SQRT_2);

    let out = ca.apply_values(|x| 0.5 * erf::erfc((mean - x) * inv_std_sqrt2));
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn normal_ppf(inputs: &[Series], kwargs: NormalKwargs) -> PolarsResult<Series> {
    let ca = extract_ca(inputs, &kwargs)?;
    let mean = kwargs.mean;
    let std_sqrt2 = kwargs.std * std::f64::consts::SQRT_2;

    let out = ca.apply(|opt_p| {
        opt_p.and_then(|p| {
            if p.is_nan() {
                Some(f64::NAN)
            } else if (0.0..=1.0).contains(&p) {
                Some(mean - std_sqrt2 * erf::erfc_inv(2.0 * p))
            } else {
                None
            }
        })
    });

    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn normal_pdf(inputs: &[Series], kwargs: NormalKwargs) -> PolarsResult<Series> {
    let ca = extract_ca(inputs, &kwargs)?;
    let mean = kwargs.mean;
    let inv_std = 1.0 / kwargs.std;
    let norm_factor = inv_std * FRAC_1_SQRT_2PI;

    let out = ca.apply_values(|x| {
        let d = (x - mean) * inv_std;
        (-0.5 * d * d).exp() * norm_factor
    });
    Ok(out.into_series())
}

#[pymodule]
fn _internal(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
