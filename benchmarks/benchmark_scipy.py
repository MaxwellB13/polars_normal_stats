import time

import numpy as np
import polars as pl
from scipy.stats import norm

from polars_normal_stats import normal_cdf, normal_pdf, normal_ppf


def benchmark_comparison(iterations=10):
    sizes = [100_000, 1_000_000, 10_000_000, 25_000_000]
    results = []
    
    print(f"Running benchmarks with {iterations} iterations...")
    print(f"{'Function':<12} | {'Size':>12} | {'SciPy (s)':>10} | {'Plugin (s)':>10} | {'Speedup':>8}")
    print("-" * 65)

    for size in sizes:
        # Data preparation
        x_np = np.random.randn(size)
        p_np = np.random.uniform(0, 1, size)
        df = pl.DataFrame({
            "x": x_np,
            "p": p_np
        })

        # --- CDF ---
        scipy_cdf_times = []
        plugin_cdf_times = []
        # Warm up
        norm.cdf(x_np)
        df.lazy().select(normal_cdf(pl.col("x"))).collect()
        
        for _ in range(iterations):
            start = time.perf_counter()
            _ = norm.cdf(x_np)
            scipy_cdf_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            df.lazy().select(normal_cdf(pl.col("x"))).collect()
            plugin_cdf_times.append(time.perf_counter() - start)
            
        scipy_cdf_mean = np.mean(scipy_cdf_times)
        plugin_cdf_mean = np.mean(plugin_cdf_times)
        results.append(["CDF", size, scipy_cdf_mean, plugin_cdf_mean, scipy_cdf_mean / plugin_cdf_mean])
        print(f"{'CDF':<12} | {size:12,} | {scipy_cdf_mean:10.4f} | {plugin_cdf_mean:10.4f} | {scipy_cdf_mean/plugin_cdf_mean:7.2f}x")

        # --- PPF ---
        scipy_ppf_times = []
        plugin_ppf_times = []
        # Warm up
        norm.ppf(p_np)
        df.lazy().select(normal_ppf(pl.col("p"))).collect()
        
        for _ in range(iterations):
            start = time.perf_counter()
            _ = norm.ppf(p_np)
            scipy_ppf_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            df.lazy().select(normal_ppf(pl.col("p"))).collect()
            plugin_ppf_times.append(time.perf_counter() - start)
            
        scipy_ppf_mean = np.mean(scipy_ppf_times)
        plugin_ppf_mean = np.mean(plugin_ppf_times)
        results.append(["PPF", size, scipy_ppf_mean, plugin_ppf_mean, scipy_ppf_mean / plugin_ppf_mean])
        print(f"{'PPF':<12} | {size:12,} | {scipy_ppf_mean:10.4f} | {plugin_ppf_mean:10.4f} | {scipy_ppf_mean/plugin_ppf_mean:7.2f}x")

        # --- PDF ---
        scipy_pdf_times = []
        plugin_pdf_times = []
        # Warm up
        norm.pdf(x_np)
        df.lazy().select(normal_pdf(pl.col("x"))).collect()
        
        for _ in range(iterations):
            start = time.perf_counter()
            _ = norm.pdf(x_np)
            scipy_pdf_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            df.lazy().select(normal_pdf(pl.col("x"))).collect()
            plugin_pdf_times.append(time.perf_counter() - start)
            
        scipy_pdf_mean = np.mean(scipy_pdf_times)
        plugin_pdf_mean = np.mean(plugin_pdf_times)
        results.append(["PDF", size, scipy_pdf_mean, plugin_pdf_mean, scipy_pdf_mean / plugin_pdf_mean])
        print(f"{'PDF':<12} | {size:12,} | {scipy_pdf_mean:10.4f} | {plugin_pdf_mean:10.4f} | {scipy_pdf_mean/plugin_pdf_mean:7.2f}x")
        print("-" * 65)

    save_as_markdown(results)

def save_as_markdown(results):
    import os
    # Use absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "comparison_results.md")
    
    # Ensure directory exists (though it should since the script is in it)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w") as f:
        f.write("# Benchmark Comparison: SciPy vs Polars Plugin\n\n")
        f.write("Results averaged over 10 iterations.\n\n")
        f.write("| Function | Size | SciPy (s) | Plugin (s) | Speedup |\n")
        f.write("| :--- | ---: | ---: | ---: | ---: |\n")
        for row in results:
            func, size, scipy_t, plugin_t, speedup = row
            f.write(f"| {func} | {size:,} | {scipy_t:.4f} | {plugin_t:.4f} | {speedup:.2f}x |\n")
    
    print(f"\nResults saved to {file_path}")

if __name__ == "__main__":
    benchmark_comparison(iterations=10)
