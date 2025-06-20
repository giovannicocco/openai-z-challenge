import numpy as np
import matplotlib.pyplot as plt

sensor_cols = [col for col in df_benchmark.columns if df_benchmark[col].dtype != object]

# Select only sensors with valid values in at least one group
valid_cols = [
    col for col in sensor_cols
    if (df_benchmark[col].notna().any() and df_candidates[col].notna().any())
]

coverage = (df_candidates[valid_cols].notna().sum() / len(df_candidates)).round(2)
print("Proportion of valid values ​​in each sensor (candidates):")
print(coverage)

z_bench = (df_benchmark[valid_cols] - df_benchmark[valid_cols].mean()) \
          / df_benchmark[valid_cols].std()

z_cand  = (df_candidates[valid_cols] - df_benchmark[valid_cols].mean()) \
          / df_benchmark[valid_cols].std()

means_bench = z_bench.mean().values
means_cand  = z_cand.mean().values

x = np.arange(len(valid_cols))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, means_bench, marker='o', label='Benchmark (z)', linewidth=2)
ax.plot(x, means_cand, marker='s', label='Candidates (z)', linewidth=2)

ax.set_ylabel("Z-score (σ)")
ax.set_title('Sensor Parameter Means Profile: Benchmarks vs. Nhamini-wi Candidates')
ax.set_xticks(x)
ax.set_xticklabels(valid_cols, rotation=45, ha='right')
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()