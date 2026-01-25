use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

fn get_scalar_param(
    inputs: &[Series],
    index: usize,
    default: f64,
    name: &str,
) -> PolarsResult<f64> {
    if inputs.len() <= index {
        return Ok(default);
    }

    let param = inputs[index].f64()?;
    if param.len() != 1 {
        return Err(PolarsError::ComputeError(
            format!("Parameter '{name}' must be a scalar value, not a column. Got length: {}", param.len()).into(),
        ));
    }

    match param.get(0) {
        Some(value) => Ok(value),
        None => Err(PolarsError::ComputeError(
            format!("Parameter '{name}' cannot be null. Please provide a valid numeric value.").into(),
        )),
    }
}

fn get_params(inputs: &[Series]) -> PolarsResult<(f64, f64)> {
    let mean = get_scalar_param(inputs, 1, 0.0, "mean")?;
    let std = get_scalar_param(inputs, 2, 1.0, "std")?;

    if std <= 0.0 {
        return Err(PolarsError::ComputeError(
            format!("Standard deviation must be positive. Got: {}", std).into(),
        ));
    }

    Ok((mean, std))
}

fn create_normal(mean: f64, std: f64) -> PolarsResult<Normal> {
    Normal::new(mean, std)
        .map_err(|e| PolarsError::ComputeError(format!("Invalid normal distribution: {}", e).into()))
}

#[polars_expr(output_type=Float64)]
fn normal_cdf(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return Err(PolarsError::ComputeError(
            "normal_cdf requires an input column".into(),
        ));
    }
    let x = inputs[0].f64()?;
    let (mean, std) = get_params(inputs)?;
    let normal = create_normal(mean, std)?;

    let out: Float64Chunked = x.apply(|opt_x| opt_x.map(|x_val| normal.cdf(x_val)));

    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn normal_ppf(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return Err(PolarsError::ComputeError(
            "normal_ppf requires an input column".into(),
        ));
    }
    let p = inputs[0].f64()?;
    let (mean, std) = get_params(inputs)?;
    let normal = create_normal(mean, std)?;

    let out: Float64Chunked = p.apply(|opt_p| {
        opt_p.and_then(|p_val| {
            if (0.0..=1.0).contains(&p_val) {
                Some(normal.inverse_cdf(p_val))
            } else {
                None
            }
        })
    });

    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn normal_pdf(inputs: &[Series]) -> PolarsResult<Series> {
    if inputs.is_empty() {
        return Err(PolarsError::ComputeError(
            "normal_pdf requires an input column".into(),
        ));
    }
    let x = inputs[0].f64()?;
    let (mean, std) = get_params(inputs)?;
    let normal = create_normal(mean, std)?;

    let out: Float64Chunked = x.apply(|opt_x| opt_x.map(|x_val| normal.pdf(x_val)));

    Ok(out.into_series())
}

#[pymodule]
fn _internal(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
