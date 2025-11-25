# 1. Introduction

## 1.1 Background
Wi-Fi access networks often accumulate dozens of access points (APs) with varying client load, CPU/memory utilization, and RF signal quality. Without a clear, data-driven grouping, it is hard to decide where to add capacity, which APs are underused, and which suffer from weak coverage. This project applies unsupervised clustering—built entirely with basic Python (pandas/numpy) and no ML libraries—to segment APs based on operational metrics. The focus is on both the clustering objective (k-means SSE) and the optimizer choices, including a deterministic annealing approach.

## 1.2 Project Scope

- Dataset: “data 3 access point” (chosen for its richest metrics: clients, CPU, memory, 2.4G/5G signals).
- Features: 13 per-AP aggregate features (client mean/max/std; CPU mean/max; memory mean/max; signal mean/min/max on 2.4G and 5G).
- Methods: two clustering strategies evaluated on the same k-means SSE objective:
  - Baseline: multi-start k-means++ + Lloyd iterations fixed at k=3, filtered for minimum cluster size.
  - Optimizer: deterministic annealing for k=3 (temperature-based soft-to-hard assignments).
- Evaluation: sum of squared errors (SSE), silhouette score, cluster sizes, and per-cluster profiles.
- Deliverables: reproducible code (`clustering.py`), result artifacts (`results_summary.json`, `results_summary.txt`), and this report.

## 1.3 Contributions

- Implemented a transparent, from-scratch clustering pipeline (no ML libraries), including feature aggregation and z-score standardization.
- Preserved all APs by mean-imputing missing metrics instead of dropping rows, expanding coverage to 287 devices.
- Built a robust baseline using multi-start k-means++ at k=3 with safeguards against tiny clusters and silhouette-based selection.
- Applied deterministic annealing as a niche optimizer for k-means SSE, yielding a 3-cluster solution with the lowest SSE.
- Produced interpretable cluster profiles (load, resource use, RF strength) to inform operational actions (capacity relief, repositioning, consolidation).
- Logged all results to JSON/TXT for traceability and downstream reporting, setting up a clear foundation for the remaining sections of this report.

## 1.4 Plain-language summary

- Goal: sort Wi-Fi access points into “hot,” “steady,” or “underused” so operators know where to add relief, monitor, or repurpose gear.
- Data: 287 APs, each with 13 long-run metrics (client load, CPU, memory, signal strength on 2.4G/5G). Missing values were filled with feature averages so no AP was discarded.
- Methods: two ways of grouping the same data—(a) many tries of standard k-means at k=3, keeping the cleanest split; (b) deterministic annealing at k=3, which starts with fuzzy groups that harden over time.
- Results: both methods yield 3 clusters; baseline separates more cleanly (higher silhouette), while annealing is tighter (lower SSE) and highlights the high-memory tier.

# 2. Dataset Description

## 2.1 Dataset Selection

We selected the “data 3 access point” dataset because it provides the richest operational context among the available options: client counts, CPU, memory, and signal metrics on both 2.4 GHz and 5 GHz bands. This breadth enables clusters that reflect both utilization and RF health, which is essential for actionable network tuning (capacity relief, coverage adjustments, or consolidation).

## 2.2 Data Summary

- Entities: 287 unique APs (all APs kept via mean imputation).
- Raw samples: millions of 5-minute measurements across five CSV files (clients, CPU, memory, 2.4G signal, 5G signal).
- Aggregation level: per-AP aggregates to capture long-run behavior rather than transient spikes.
- Feature set (13 per AP):
  - Clients: mean, max, std of `client_count`.
  - CPU: mean, max of `cpu_usage_ratio`.
  - Memory: mean, max of `memory_usage_ratio`.
  - 2.4G signal (dBm): mean, min, max of `signal_dbm`.
  - 5G signal (dBm): mean, min, max of `signal_dbm`.

## 2.3 Preprocessing Steps

- Loading: used pandas to read the five gzipped CSVs for clients, CPU, memory, 2.4G signal, and 5G signal.
- Aggregation: grouped by `ap_name` and computed the 13 statistics listed above. For client std, any NaN (constant series) was set to zero to avoid dropping deterministic rows.
- Imputation: retained all APs by mean-imputing any missing metric per feature (no rows dropped); final shape is 287 APs × 13 features.
- Scaling: applied z-score standardization (mean=0, std=1 per feature) so no single metric dominates Euclidean distances in the clustering objective.
- Integrity checks: verified row/column counts after aggregation, ensured no remaining NaNs post-merge, and preserved AP names for later cluster interpretation.


# 3. Clustering Method

## 3.1 Choice of Algorithm

We use the k-means objective (sum of squared errors, SSE) because:
- It is simple, interpretable, and directly ties to “tightness” of clusters.
- It pairs naturally with Euclidean distance on standardized features.
- It allows clear optimization experiments (seeded k-means++ vs. deterministic annealing) without external ML libraries.

Plain terms for the two scores we report:
- SSE (sum of squared errors): how tightly points sit inside their assigned cluster. Lower is better (tighter groups).
- Silhouette: how far points are from other clusters relative to their own. Higher is better (cleaner separation).

Two strategies optimize the same objective:
- Baseline: multi-start k-means++ + Lloyd iterations at k=3 with 120 restarts, selecting the run with the highest silhouette (tie-break by lowest SSE).
- Optimizer: deterministic annealing for k=3, a temperature-driven, soft-to-hard assignment method that is more niche and course-relevant.

## 3.2 Mathematical Formulation

Given data matrix \( X \in \mathbb{R}^{n \times d} \) and k centroids \( C \in \mathbb{R}^{k \times d} \):
- Assignment: each point \( x_i \) is assigned to the closest centroid by Euclidean distance.
- Objective (SSE): \( \text{SSE} = \sum_i \| x_i - c_{a(i)} \|_2^2 \), where \( a(i) \) is the assigned cluster for point i.
- Silhouette (for evaluation): for each point, \( s = (b - a) / \max(a, b) \), where \( a \) is mean intra-cluster distance and \( b \) is the smallest mean distance to another cluster; overall silhouette is the mean of s.

## 3.3 Implementation Details

- Distance and objective: implemented manually with numpy; no external ML libraries.
- Standardization: z-scores per feature prior to clustering to balance scales.
- Baseline procedure:
  - k-means++ seeding to spread initial centroids.
  - Lloyd updates until centroid shift < 1e-4 or 100 iterations.
  - Fixed k=3 with 120 restarts; discard solutions with any cluster size < 5.
  - Select by highest silhouette; tie-break by lowest SSE.
- Deterministic annealing procedure:
  - Start with random centroids (k=3).
  - Soft assignments weighted by \( \exp(-d^2 / T) \); lower temperatures harden assignments.
  - Cooling schedule: \( T \) from 6.0 down to 0.01, multiplicative factor 0.9, with 12 refinement steps per temperature.
  - Track best centroids/SSE across the cooling path; finalize with hard assignments and compute silhouette.


# 4. Optimization Method

## 4.1 Rationale for Choosing the Selected Method

Deterministic annealing clustering (DAC) is chosen as the niche optimizer because:
- It introduces a temperature-controlled soft assignment, which helps escape poor local minima typical of k-means.
- It fits the course theme of optimization methods beyond standard heuristics.
- It maintains the same SSE objective, making results directly comparable to the baseline k-means++ approach.

## 4.2 Mathematical Formulation

- Soft assignment weights at temperature \( T \):
  \( w_{ij} = \frac{\exp(-\|x_i - c_j\|^2 / T)}{\sum_{l} \exp(-\|x_i - c_l\|^2 / T)} \).
- Centroid update:
  \( c_j = \frac{\sum_i w_{ij} x_i}{\sum_i w_{ij}} \).
- Cooling schedule:
  Start \( T_0 = 6.0 \), end \( T_{\min} = 0.01 \), multiply by \( \alpha = 0.9 \) each outer step, with multiple inner refinement steps per temperature.
- Final hard assignment and SSE:
  After cooling, assign each point to its nearest centroid; compute SSE and silhouette.

## 4.3 Implementation Details

- Initialization: random selection of k=3 points as initial centroids.
- Soft update loop: at each temperature, compute squared distances, derive soft weights, and update centroids with weighted averages.
- Cooling: \( T \leftarrow \alpha T \) after each set of inner refinement steps (12 per temperature level).
- Tracking best solution: keep the lowest-SSE centroids encountered over the full cooling schedule to guard against late-stage degradation.
- Evaluation: once the best centroids are identified, compute hard labels, SSE, silhouette, and cluster sizes; record the temperature/SSE history for convergence inspection.

# 5. Experimental Setup

## 5.1 Hardware and Software
- Environment: Python 3 with pandas and numpy only (no external ML libraries).
- Hardware: standard CPU environment; clustering runs quickly on 287 APs × 13 features.
- Reproducibility: seeds are fixed (k-means++ seed=99; annealing seed=123) for consistent results.

## 5.2 Hyperparameters
- Standardization: z-scores per feature.
- Baseline (multi-start k-means++):
  - k: 3 (fixed).
  - Restarts: 120.
  - Minimum cluster size: 5 (to avoid degenerate tiny clusters).
  - Lloyd tolerance: centroid shift < 1e-4 or 100 iterations.
- Optimizer (deterministic annealing):
  - k=3.
  - Initial temperature \( T_0 = 6.0 \), minimum temperature \( T_{\min} = 0.01 \).
  - Cooling factor \( \alpha = 0.9 \).
  - Inner refinement steps per temperature: 12.
  - Seed: 123.

## 5.3 Baseline
- Objective: same SSE objective as the optimizer for apples-to-apples comparison.
- Procedure: for each restart at k=3, run k-means++ seeding + Lloyd updates; reject any solution with a cluster smaller than 5 points; evaluate SSE and silhouette; keep the solution with highest silhouette (tie-break by lowest SSE).
- Purpose: establish a strong, multi-start baseline and a silhouette-driven choice without varying k, keeping capacity equal to annealing for a fairer comparison.

# 6. Results and Evaluation

In plain English: with k fixed at 3, the baseline separates clusters more cleanly (higher silhouette), while deterministic annealing packs them tighter (lower SSE) and draws out a distinct high-memory tier. Lower SSE = tighter clusters; higher silhouette = clearer separation.

## 6.1 Baseline Clustering Results
- Method: multi-start k-means++ + Lloyd, k=3, min cluster size 5, 120 restarts.
- Metrics: SSE = 2803.64, silhouette = 0.212, sizes = [86, 25, 176].
- Cluster profiles:
  - Cluster 0 (86 APs): clients_mean ≈ 3.22, CPU ≈ 0.056, memory ≈ 0.458, signals ≈ -63/-58 dBm.
  - Cluster 1 (25 APs): clients_mean ≈ 2.17, CPU ≈ 0.031, memory ≈ 0.462, signals ≈ -59/-58 dBm.
  - Cluster 2 (176 APs): clients_mean ≈ 0.69, CPU ≈ 0.019, memory ≈ 0.456, signals ≈ -65/-58 dBm.

## 6.2 Optimized Clustering Results
- Method: deterministic annealing clustering (k=3).
- Metrics: SSE = 2731.39, silhouette = 0.166, sizes = [88, 82, 117].
- Cluster profiles:
  - Cluster 0 (88 APs): clients_mean ≈ 3.38, CPU ≈ 0.056, memory ≈ 0.457, signals ≈ -63/-57 dBm.
  - Cluster 1 (82 APs): clients_mean ≈ 1.04, CPU ≈ 0.023, memory ≈ 0.570, signals ≈ -63/-55 dBm.
  - Cluster 2 (117 APs): clients_mean ≈ 0.60, CPU ≈ 0.019, memory ≈ 0.378, signals ≈ -65/-61 dBm.

## 6.3 Quantitative Metrics
- Baseline: SSE 2803.64; silhouette 0.212; k=3; sizes [86, 25, 176].
- Annealing: SSE 2731.39; silhouette 0.166; k=3; sizes [88, 82, 117].
- SSE vs. silhouette trade-off:
  - Baseline wins on separation (higher silhouette) and keeps hot vs. underused clearer.
  - Annealing wins on within-cluster tightness (lower SSE) and isolates a pronounced high-memory tier, at the cost of overlap.

## 6.4 Comparison Table

| Method                         | k  | SSE     | Silhouette | Sizes           | Notes                                    |
|--------------------------------|----|---------|------------|-----------------|------------------------------------------|
| Baseline: MS k-means++ + Lloyd | 3  | 2803.64 | 0.212      | [86, 25, 176]   | Better separation; clearer hot vs. underused |
| Deterministic Annealing        | 3  | 2731.39 | 0.166      | [88, 82, 117]   | Lower SSE; distinct high-memory tier        |

# 7. Discussion

## 7.1 Why the Clustering Turned Out This Way
- Load and memory drive separation in both methods: hot APs (Clusters 0) show the highest client load and CPU; underused APs (Clusters 2) carry very low clients; a third tier carries moderate load with either stronger signals (baseline) or higher memory (annealing).
- Signal quality reinforces tiers: underused APs sit around -65/-61 dBm, the hot tier around -63/-57 dBm, and the high-memory/strong-signal tiers closer to -63/-55 dBm. Weaker signals align with underuse; stronger signals align with hotter or memory-heavy tiers.

## 7.2 Impact of Optimization
- Baseline (MS k-means++): at fixed k=3, broad seeding plus restarts maximizes silhouette (0.212), giving clearer separation between hot and underused groups.
- Deterministic annealing: at the same k=3, lowers SSE further (2731.39) and sharpens the high-memory tier, but silhouette drops to 0.166. The cooling path still improves tightness versus a single k-means run.

## 7.3 Limitations
- Metric scope: no explicit interference/channel utilization metrics; temporal dynamics are aggregated away, so bursts are not captured.
- Model scope: Euclidean k-means assumes roughly spherical clusters; non-spherical structure is not modeled.
- Trade-off at fixed k: baseline labels separate more cleanly (higher silhouette), annealing labels are tighter (lower SSE) but overlap more.
- Stability beyond seeds: seeds are fixed; broader stability checks (e.g., multiple seeds, slight perturbations) could further validate robustness.

# 8. Conclusion
We built a transparent, from-scratch clustering pipeline on the “data 3 access point” dataset, comparing a strong baseline (multi-start k-means++ + Lloyd) and a niche optimizer (deterministic annealing), both at k=3. After keeping all 287 APs via mean imputation, the baseline delivered the best separation (silhouette 0.212) while deterministic annealing achieved the lowest SSE (2731.39), carving out a clear high-memory tier. Depending on operational needs, pick baseline labels for clearer separation or annealing labels for tighter clusters and memory-focused targeting. Future work: add interference/channel features, test stability across seeds/perturbations, and adjust k if actions demand finer granularity.
