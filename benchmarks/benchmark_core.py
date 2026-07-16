import time

import numpy as np
import polars as pl

from polars_normal_stats import normal_cdf, normal_pdf, normal_ppf


def run_benchmark():
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    functions = {
        "normal_cdf": normal_cdf,
        "normal_ppf": normal_ppf,
        "normal_pdf": normal_pdf
    }

    results = []

    for size in sizes:
        print(f"Benchmarking size: {size:,}")
        # Generate random data
        # For cdf/pdf we use normal values, for ppf we use probabilities [0, 1]
        df = pl.DataFrame({
            "x": np.random.randn(size),
            "p": np.random.uniform(0, 1, size)
        })

        size_results = {"size": size}
        
        for name, func in functions.items():
            col = "p" if name == "normal_ppf" else "x"
            
            # Warm up
            df.lazy().with_columns(res=func(pl.col(col))).collect()
            
            start_time = time.perf_counter()
            df.lazy().with_columns(res=func(pl.col(col))).collect()
            end_time = time.perf_counter()
            
            duration = end_time - start_time
            size_results[name] = duration
            print(f"  {name}: {duration:.4f}s")
        
        results.append(size_results)

    # Print summary table
    print("\nBenchmark Results (seconds):")
    print(f"{'Rows':>12} | {'CDF':>10} | {'PPF':>10} | {'PDF':>10}")
    print("-" * 50)
    for res in results:
        print(f"{res['size']:12,} | {res['normal_cdf']:10.4f} | {res['normal_ppf']:10.4f} | {res['normal_pdf']:10.4f}")

    return results

if __name__ == "__main__":
    run_benchmark()
