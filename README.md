# Polars Normal Stats

Fast normal distribution functions (CDF, PPF, PDF) for Polars DataFrames, implemented as a Polars plugin in Rust.

This plugin provides highly optimized implementations of the Normal (Gaussian) distribution functions, offering significant speedups over calling SciPy's `norm` functions within a Polars `map_batches` or `apply` (now `map_elements`).

## Features

- **normal_cdf(x, mean=0.0, std=1.0)**: Cumulative Distribution Function.
- **normal_ppf(p, mean=0.0, std=1.0)**: Percent Point Function (Inverse CDF).
- **normal_pdf(x, mean=0.0, std=1.0)**: Probability Density Function.
- Fully compatible with Polars' **lazy execution** and expression API.
- Optimized using Rust `kwargs` for distribution parameters.

## Installation

Install using `uv`:
```bash
uv add polars-normal-stats
```

Install using `pip`:
```bash
pip install polars-normal-stats
```

*(Note: Ensure you have `polars` installed as well.)*

## Usage

The functions are designed to work directly within Polars expressions.

```python
import polars as pl
from polars_normal_stats import normal_cdf, normal_ppf, normal_pdf

df = pl.DataFrame({
    "x": [-1.0, 0.0, 1.0],
    "p": [0.1, 0.5, 0.9]
})

result = df.select([
    normal_cdf(pl.col("x")).alias("cdf"),
    normal_ppf(pl.col("p"), mean=10.0, std=2.0).alias("ppf_shifted"),
    normal_pdf(pl.col("x"), mean=0.0, std=1.0).alias("pdf")
])

print(result)
```

### Lazy Execution

Since these functions return Polars expressions, they integrate seamlessly into Polars' lazy API. This allows Polars to optimize the entire query plan, including these statistical operations.

```python
lazy_result = (
    pl.scan_parquet("data.parquet")
    .with_columns(
        z_score = normal_cdf(pl.col("value"), mean=100.0, std=15.0)
    )
    .collect()
)
```

## Benchmarks

The plugin is significantly faster than using SciPy's normal distribution functions via Polars' `map_batches`. Below are the results comparing the execution time for varying data sizes.

Results averaged over 10 iterations:

| Function | Size | SciPy (s) | Plugin (s) | Speedup |
| :--- | ---: | ---: | ---: | ---: |
| CDF | 100,000 | 0.0025 | 0.0019 | 1.29x |
| PPF | 100,000 | 0.0035 | 0.0016 | 2.23x |
| PDF | 100,000 | 0.0018 | 0.0006 | 2.86x |
| CDF | 1,000,000 | 0.0256 | 0.0191 | 1.34x |
| PPF | 1,000,000 | 0.0355 | 0.0147 | 2.42x |
| PDF | 1,000,000 | 0.0234 | 0.0064 | 3.65x |
| CDF | 10,000,000 | 0.2702 | 0.1903 | 1.42x |
| PPF | 10,000,000 | 0.3637 | 0.1436 | 2.53x |
| PDF | 10,000,000 | 0.2520 | 0.0604 | 4.17x |
| CDF | 25,000,000 | 0.6841 | 0.4680 | 1.46x |
| PPF | 25,000,000 | 0.9122 | 0.3587 | 2.54x |
| PDF | 25,000,000 | 0.6424 | 0.1506 | 4.27x |

*Benchmarks performed on 10,000,000+ rows show up to a **4.2x speedup** for PDF calculations.*

## Credits

This plugin was developed using the excellent [polars-xdt](https://github.com/MarcoGorelli/polars-xdt) as a template and acknowledges the work of [Marco Gorelli](https://github.com/MarcoGorelli), [Ritchie Vink](https://github.com/ritchie46), and the Polars contributors for making Python-Rust plugin development accessible.

It also relies on the [statrs](https://github.com/statrs-dev/statrs) crate for statistical computations and [PyO3](https://github.com/PyO3/pyo3) for Rust-Python bindings.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
