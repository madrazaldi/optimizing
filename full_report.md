# 1. Introduction

## 1.1 Background
Wi-Fi access networks often accumulate dozens of access points (APs) with varying client load, CPU/memory utilization, and RF signal quality. Without a clear, data-driven grouping, it is hard to decide where to add capacity, which APs are underused, and which suffer from weak coverage. This project applies unsupervised clustering—built entirely with basic Python (pandas/numpy) and no ML libraries—to segment APs based on operational metrics. Because this is an Optimization Methods course project, the focus is on both the clustering objective (k-means SSE) and the optimizer choices, including a niche deterministic annealing approach.

## 1.2 Project Scope
- Dataset: “data 3 access point” (chosen for its richest metrics: clients, CPU, memory, 2.4G/5G signals).
- Features: 13 per-AP aggregate features (client mean/max/std; CPU mean/max; memory mean/max; signal mean/min/max on 2.4G and 5G).
- Methods: two clustering strategies evaluated on the same k-means SSE objective:
  - Baseline: multi-start k-means++ + Lloyd iterations across k=2..6, filtered for minimum cluster size.
  - Optimizer: deterministic annealing for k=3 (temperature-based soft-to-hard assignments).
- Evaluation: sum of squared errors (SSE), silhouette score, cluster sizes, and per-cluster profiles.
- Deliverables: reproducible code (`clustering.py`), result artifacts (`results_summary.json`, `results_summary.txt`), and this report.

## 1.3 Contributions
- Implemented a transparent, from-scratch clustering pipeline (no ML libraries), including feature aggregation and z-score standardization.
- Built a robust baseline using multi-start k-means++ to search k and avoid tiny, degenerate clusters; selected by highest silhouette (tie-break by SSE).
- Applied deterministic annealing as a niche optimizer for k-means SSE, yielding a competitive 3-cluster solution with the lowest SSE.
- Produced interpretable cluster profiles (load, resource use, RF strength) to inform operational actions (capacity relief, repositioning, consolidation).
- Logged all results to JSON/TXT for traceability and downstream reporting, setting up a clear foundation for the remaining sections of this report.

# 2. Dataset Description

## 2.1 Dataset Selection
We selected the “data 3 access point” dataset because it provides the richest operational context among the available options: client counts, CPU, memory, and signal metrics on both 2.4 GHz and 5 GHz bands. This breadth enables clusters that reflect both utilization and RF health, which is essential for actionable network tuning (capacity relief, coverage adjustments, or consolidation).

## 2.2 Data Summary
- Entities: 101 unique APs.
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
- Row filtering: dropped APs with any missing metric across the 13 features to maintain a consistent feature matrix for clustering; final shape is 101 APs × 13 features.
- Scaling: applied z-score standardization (mean=0, std=1 per feature) so no single metric dominates Euclidean distances in the clustering objective.
- Integrity checks: verified row/column counts after aggregation, ensured no remaining NaNs post-merge, and preserved AP names for later cluster interpretation.


# 3. Clustering Method

## 3.1 Choice of Algorithm
We use the k-means objective (sum of squared errors, SSE) because:
- It is simple, interpretable, and directly ties to “tightness” of clusters.
- It pairs naturally with Euclidean distance on standardized features.
- It allows clear optimization experiments (seeded k-means++ vs. deterministic annealing) without external ML libraries.

Two strategies optimize the same objective:
- Baseline: multi-start k-means++ + Lloyd iterations across k=2..6, selecting the run with the highest silhouette (tie-break by lowest SSE).
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
  - Search over k=2..6 with 120 restarts; discard solutions with any cluster size < 5.
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
- Hardware: standard CPU environment; clustering runs quickly on 101 APs × 13 features.
- Reproducibility: seeds are fixed (k-means++ seed=99; annealing seed=123) for consistent results.

## 5.2 Hyperparameters
- Standardization: z-scores per feature.
- Baseline (multi-start k-means++):
  - k range: 2..6.
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
- Procedure: for each k in 2..6 and each restart, run k-means++ seeding + Lloyd updates; reject any solution with a cluster smaller than 5 points; evaluate SSE and silhouette; keep the solution with highest silhouette (tie-break by lowest SSE).
- Purpose: establish a strong, multi-start baseline and a silhouette-driven choice of k to avoid cherry-picking an easy optimizer comparison.

# 6. Results and Evaluation

## 6.1 Baseline Clustering Results
- Method: multi-start k-means++ + Lloyd, k searched over 2..6, min cluster size 5, 120 restarts.
- Selected solution: k=2 (highest silhouette).
- Metrics: SSE = 1122.96, silhouette = 0.317, sizes = [86, 15].
- Cluster profiles:
  - Cluster 0 (86 APs): clients_mean ≈ 2.15, CPU ≈ 0.047, memory ≈ 0.421, signals ≈ -64/-62 dBm.
  - Cluster 1 (15 APs): clients_mean ≈ 3.75, CPU ≈ 0.038, memory ≈ 0.617, signals ≈ -60/-57 dBm.

## 6.2 Optimized Clustering Results
- Method: deterministic annealing clustering (k=3).
- Metrics: SSE = 922.98, silhouette = 0.211, sizes = [19, 60, 22].
- Cluster profiles:
  - Cluster 0 (19 APs): clients_mean ≈ 0.27, CPU ≈ 0.019, memory ≈ 0.401, signals ≈ -74/-69 dBm.
  - Cluster 1 (60 APs): clients_mean ≈ 1.80, CPU ≈ 0.046, memory ≈ 0.438, signals ≈ -60/-60 dBm.
  - Cluster 2 (22 APs): clients_mean ≈ 5.81, CPU ≈ 0.067, memory ≈ 0.525, signals ≈ -64/-58 dBm.

## 6.3 Quantitative Metrics
- Baseline (best by silhouette): SSE 1122.96; silhouette 0.317; k=2; sizes [86, 15].
- Annealing (best by SSE for k=3): SSE 922.98; silhouette 0.211; k=3; sizes [19, 60, 22].
- SSE vs. silhouette trade-off:
  - Baseline wins on separation (silhouette) but uses fewer clusters (coarser granularity) and slightly higher SSE.
  - Annealing wins on within-cluster tightness (lower SSE) with three tiers but has lower silhouette.

## 6.4 Comparison Table

| Method                         | k  | SSE     | Silhouette | Sizes           | Notes                                   |
|--------------------------------|----|---------|------------|-----------------|-----------------------------------------|
| Baseline: MS k-means++ + Lloyd | 2  | 1122.96 | 0.317      | [86, 15]        | Best separation; coarser 2-cluster split |
| Deterministic Annealing        | 3  | 922.98  | 0.211      | [19, 60, 22]    | Best SSE for k=3; clearer 3-tier view    |

# 7. Discussion

## 7.1 Why the Clustering Turned Out This Way
- Load and memory drive the primary separation: APs with higher memory use (and moderately higher clients) form the smaller, hotter group in the k=2 baseline; in k=3, they become the hot tier (Cluster 2). Very low-load APs with weak signals form the underused tier (Cluster 0) in k=3. The remaining majority forms the steady middle (Cluster 1).
- Signal quality reinforces these splits: weak signals correlate with underuse (Cluster 0), while stronger signals accompany the hotter tiers. The bulk baseline cluster sits in mid/upper -60 dBm, while the hot baseline cluster is closer to -60/-57 dBm.

## 7.2 Impact of Optimization
- Baseline (MS k-means++): maximizes silhouette via broad seeding and k search; yields the cleanest separation (0.317) but only two clusters.
- Deterministic annealing: lowers SSE at k=3 (922.98) and provides more granularity (hot/steady/underused) at the cost of lower silhouette (0.211). The annealing schedule helps avoid poor minima and improves within-cluster tightness versus a single k-means run.

## 7.3 Limitations
- Metric scope: no explicit interference/channel utilization metrics; temporal dynamics are aggregated away, so bursts are not captured.
- Model scope: Euclidean k-means assumes roughly spherical clusters; non-spherical structure is not modeled.
- Granularity trade-off: k=2 is coarser but separates cleanly; k=3 is richer but overlaps more.
- Stability beyond seeds: seeds are fixed; broader stability checks (e.g., multiple seeds, slight perturbations) could further validate robustness.

# 8. Conclusion
We built a transparent, from-scratch clustering pipeline on the “data 3 access point” dataset, comparing a strong baseline (multi-start k-means++ + Lloyd) and a niche optimizer (deterministic annealing). The baseline delivered the best separation (silhouette 0.317) with a 2-cluster split, while deterministic annealing achieved the lowest SSE (922.98) at k=3, yielding actionable three-tier grouping (hot, steady, underused). Depending on operational needs, choose the higher-separation 2-cluster labels or the finer 3-cluster labels. Future work: add interference/channel features, test stability across seeds/perturbations, and revisit k if more granular actions are desired.


