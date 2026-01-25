use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

#[derive(Deserialize)]
struct NormalKwargs {
    mean: f64,
    std: f64,
}

fn create_normal(mean: f64, std: f64) -> PolarsResult<Normal> {
    if std <= 0.0 {
        return Err(PolarsError::ComputeError(
            format!("Standard deviation must be positive. Got: {}", std).into(),
        ));
    }
    Normal::new(mean, std)
        .map_err(|e| PolarsError::ComputeError(format!("Invalid normal distribution: {}", e).into()))
}

#[polars_expr(output_type=Float64)]
fn normal_cdf(inputs: &[Series], kwargs: NormalKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let normal = create_normal(kwargs.mean, kwargs.std)?;

    let out = ca.apply_values(|x| normal.cdf(x));
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn normal_ppf(inputs: &[Series], kwargs: NormalKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let normal = create_normal(kwargs.mean, kwargs.std)?;

    let out = ca.apply(|opt_p| {
        opt_p.and_then(|p| {
            if (0.0..=1.0).contains(&p) {
                Some(normal.inverse_cdf(p))
            } else {
                None
            }
        })
    });

    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn normal_pdf(inputs: &[Series], kwargs: NormalKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let normal = create_normal(kwargs.mean, kwargs.std)?;

    let out = ca.apply_values(|x| normal.pdf(x));
    Ok(out.into_series())
}

#[pymodule]
fn _internal(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
