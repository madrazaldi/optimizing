# Access Point Clustering Report

This report explains, in plain language, how we grouped Wi-Fi access points (APs) using only basic Python (numpy/pandas/matplotlib) and three hand-built clustering/optimization methods. The goal was to find meaningful groups (clusters) of APs based on their load, resource usage, and signal quality—without relying on machine-learning libraries. Below you will find detailed data preparation steps, algorithm descriptions, evaluation metrics, results, operational interpretations, stability checks, limitations, and next-step recommendations.

## What data we used and why
- Chosen dataset: **data 3 access point** because it contains the richest operational context (client load, CPU, memory, and signal on both 2.4 GHz and 5 GHz). This allows clusters to reflect both utilization and RF quality.
- Sampling: each AP has many 5-minute samples over time. We aggregate per-AP statistics to keep the focus on long-run behavior rather than individual time slices.
- Feature set (13 per AP):
  - Client load: mean, max, and std of connected clients.
  - CPU usage ratio: mean and max.
  - Memory usage ratio: mean and max.
  - Signal strength (dBm) on 2.4G: mean, min, max.
  - Signal strength (dBm) on 5G: mean, min, max.
- Cleaning: drop APs with any missing metric across these 13 features. Final dataset: **101 APs x 13 features**. Keeping AP names lets us map clusters back to infrastructure.
- Rationale: These features capture demand (clients), resource stress (CPU/memory), and RF conditions (signal). Together they can separate overloaded vs idle vs weak-signal APs.

## How we prepared the data
- Standardization: z-score each feature (subtract mean, divide by std) so all dimensions contribute fairly; prevents high-variance metrics from dominating distance calculations.
- Index preservation: AP names stay aligned with rows to map cluster labels back to devices.
- Visualization aid: a lightweight PCA (via numpy SVD) projects to 2D for plotting only; clustering is performed in the full standardized feature space.
- Sanity checks performed:
  - Verified row/column counts after aggregation.
  - Replaced NaN std of client counts (for constant series) with zero to keep deterministic scaling.
  - Confirmed no remaining NaNs before clustering.

## What “clustering” means here
- Goal: APs in the same cluster should be similar; APs in different clusters should differ meaningfully on load, resource, and RF metrics.
- Distance: Euclidean distance in standardized space; squared distances feed the objective.
- Objective: minimize total sum of squared errors (SSE) between points and their assigned centroids. Lower SSE means tighter clusters.
- Separation metric: maximize silhouette (range -1..1) to encourage between-cluster distance vs within-cluster cohesion.
- Implementation: all objectives and optimizers are hand-coded (no ML libraries) to keep the pipeline transparent and reproducible.

## Three clustering approaches we tried
We used three strategies to find good centroids and compared them on SSE and silhouette:

1) **Deterministic Annealing Clustering (DAC) — primary choice**
   - Think of it as “soft k-means with a temperature knob.” At high temperature, every point is softly assigned to all clusters (broad exploration). As temperature cools, assignments harden to the closest centroid (focused refinement).
   - Settings: start temperature T0=6.0, end Tmin=0.01, cooling rate α=0.9, 12 refinement steps per temperature level. We track the best centroids during the entire cooling schedule.
   - Why it’s good: annealing helps avoid bad local minima and often finds lower SSE than a single hard-assignment run.

2) **Flower Pollination Algorithm (FPA) — baseline metaheuristic**
   - Inspired by pollination; mixes global “Lévy flight” steps (big jumps to explore) with local mixing steps (small refinements).
   - Settings: β=1.5 (Lévy), population=25 candidate solutions, 120 iterations, global pollination probability=0.85. Centroids are clipped to the observed feature range.
   - Why it’s included: it’s a relatively niche heuristic and serves as a comparison point.

3) **Multi-start k-means++ (MS k-means++) — new best-by-silhouette sweep**
   - Re-seeds k-means with k-means++ across k=2..6, keeps only solutions where every cluster has at least 5 APs, and selects the highest silhouette (tie-break by lower SSE).
   - Settings: 120 restarts, k range 2–6, min cluster size 5, seed=99.
   - Why it’s good: systematic seeding plus a silhouette-first objective avoids tiny outlier clusters and surfaces cleaner separation on this dataset.

## How we judged the results
- **SSE (sum of squared errors):** tightness metric; lower is better.
- **Silhouette score (manual implementation):** separation metric; higher is better.
- **Cluster size balance:** reject degenerate solutions with tiny clusters (MS k-means++ enforces min size 5).
- **Profiles:** per-cluster means on key metrics (clients_mean, clients_max, CPU, memory, signals) to ensure operational interpretability.
- **Plots (when run interactively):**
  - PCA scatter colored by cluster for intuition on separation.
  - FPA convergence curve to show early vs late improvements.
  - DAC SSE vs temperature to show annealing path quality.
- **Decision rule for “best method”:** choose the model with the highest silhouette; if tied, prefer lower SSE.

## Results (numbers and what they mean)
- **MS k-means++ (best silhouette):** silhouette **0.317** at **k=2**, sizes **[86, 15]**, SSE 1123.0.
- **DAC (best SSE among k=3):** SSE **923.0**, silhouette **0.211**, cluster sizes **[60, 22, 19]** APs.
- **FPA (baseline k=3):** SSE 1094.8, silhouette 0.200, cluster sizes [75, 16, 10] APs.
- Takeaway: MS k-means++ gives the clearest separation (higher silhouette) but with a coarser 2-cluster split; DAC gives the tightest 3-cluster fit (lowest SSE) if you want three tiers; FPA is the weakest of the three on both metrics. The project files tag MS k-means++ as `best_method` because silhouette is prioritized over SSE.

## What the clusters look like (plain-English profiles)
- **MS k-means++ (k=2) split:** 
  - **Cluster M0 (86 APs):** steady to moderately loaded APs; clients_mean ≈ 2.15, CPU ≈ 0.047, memory ≈ 0.421, signals around -64/-62 dBm. Interpretation: the bulk “normal” fleet—healthy, moderate utilization, baseline RF quality.
  - **Cluster M1 (15 APs):** heavier memory use and stronger signals; clients_mean ≈ 3.75, memory ≈ 0.617. Interpretation: relatively hotter APs—fewer units carrying higher resource load; candidates for capacity checks and potential relief.
- **DAC (k=3) split:** 
  - **Cluster A (60 APs):** “steady workhorses” — clients_mean ≈ 1.80, CPU ≈ 0.046, memory ≈ 0.438, signals around -60 dBm. Interpretation: mainline healthy APs.
  - **Cluster B (22 APs):** “hot/priority APs” — clients_mean ≈ 5.81, CPU ≈ 0.067, memory ≈ 0.525, signals ≈ -64 dBm (2.4G) / -58 dBm (5G). Interpretation: busiest APs; monitor for channel/coverage/capacity.
  - **Cluster C (19 APs):** “underused, weak-signal APs” — clients_mean ≈ 0.27, CPU ≈ 0.019, memory ≈ 0.401, signals roughly -74 dBm (2.4G) / -69 dBm (5G). Interpretation: likely peripheral/overprovisioned or poorly placed APs; candidates for repositioning or consolidation.

## Why the clusters look this way
- **Load + memory** are the primary separators: busy APs pull together; very idle APs pull together; the rest form the middle. Memory usage is especially discriminative for the hotter group.
- **Signal quality** provides secondary separation: weak-signal, idle APs cluster together (DAC Cluster C), while stronger-signal APs with real traffic sit in the busier clusters.
- **MS k-means++ vs DAC/FPA:** many seeded runs plus silhouette-first selection finds a simpler (k=2) split with cleaner between-cluster distance; DAC remains best for a 3-way, lower-SSE partition if you prefer finer granularity; FPA trails both on both metrics.
- **Balance vs separation:** enforcing min cluster size in MS k-means++ prevents degenerate one-point clusters, so the silhouette gain reflects genuine separation rather than outliers.

## How to read the plots
- **PCA scatter:** Each point is an AP projected to 2D for visualization. Colors are clusters. With MS k-means++ (k=2) the silhouette jumps to ~0.32 (cleaner split); with DAC (k=3) it’s ~0.21 (moderate but actionable separation). Expect some overlap because PCA compresses 13D to 2D.
- **FPA convergence:** Shows SSE dropping quickly at first, then flattening around 1095—evidence of early gains then stagnation.
- **DAC SSE vs Temperature:** SSE shrinks as temperature cools, hitting ~923 at low temperature—evidence the annealing schedule found a better basin.

## Practical actions this enables
- **If you want the clearest separation:** use the MS k-means++ labels (k=2). Focus on the 15 APs in cluster M1 for load balancing/capacity planning; M0 is your baseline.
- **If you want three actionable tiers:** stick with DAC labels. Focus on DAC Cluster B (hot) for channel/capacity tuning, clean up DAC Cluster C (weak/idle) via repositioning/retuning/consolidation, keep Cluster A as baseline.
- **Operational checks:** for hot clusters, review channel overlap, backhaul, and client distribution; for weak-signal/idle clusters, check placement, antenna orientation, and potential removal if redundant.
- Keep monitoring for drift: APs migrating from steady → hot or from steady → weak/idle signal where possible repositioning is needed. Periodically re-run clustering after changes to see movement between clusters.

## Stability, limitations, and next steps
- **Stability:** MS k-means++ uses many restarts and a min-cluster-size filter; results are stable under the given seed. DAC is also stable under repeated runs with the same schedule. FPA shows more variance but is not the selected best.
- **Metric limitations:** silhouette favors separation, SSE favors tightness; neither captures temporal bursts or interference explicitly. A 2-cluster model simplifies nuance; a 3-cluster model offers more operational tiers.
- **Feature limitations:** no direct interference/channel quality metrics; adding them could sharpen separation. Traffic mix (e.g., application categories) is not represented.
- **Next analytical steps:** (1) broaden k search or adjust min-cluster-size if you want finer splits; (2) add features on retransmissions or channel utilization if available; (3) validate cluster stability by perturbing seeds and checking label consistency; (4) overlay geography to see spatial patterns.

## Takeaway
- The new multi-start k-means++ sweep delivers the best silhouette (0.317) with a clean two-cluster split; DAC still gives the tightest three-cluster fit (lowest SSE) if you need three tiers. Use whichever labeling matches your operational granularity, but both outperform the original FPA baseline. Re-run after network changes to track how APs migrate between clusters.

## Files
- Code and analysis: `clustering.ipynb` (includes FPA, DAC, and MS k-means++; run all cells to reproduce).
- Outputs: `results_summary.json`, `results_summary.txt`.
