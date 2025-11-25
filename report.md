# Clustering Project Presentation Outline

- **Problem & setup**: Sort Wi-Fi access points (APs) into useful buckets—hot, steady, underused—using only pandas/numpy. Keep every AP by filling missing metrics with the feature’s average.
- **Data (what we fed the model)**: “data 3 access point”; five CSVs of AP stats → 13 features per AP (client load, CPU, memory, 2.4G/5G signal strength). After filling gaps: 287 APs × 13 features, standardized so no metric dominates.
- **Methods compared (plain English)**
  - Baseline: run k-means++ many times at k=3 and keep the split with the best “separation score.” Forces each cluster to have at least 5 APs.
  - Optimizer: deterministic annealing—start with soft groupings that harden over time; also at k=3.
- **Key metrics (how well the splits hold together)**
  - Baseline (k=3): SSE 2803.64 (lower is tighter), silhouette 0.212 (higher is cleaner), sizes [86, 25, 176].
  - Annealing (k=3): SSE 2731.39, silhouette 0.166, sizes [88, 82, 117].
- **Cluster narratives (what the groups mean)**
  - Baseline k=3: Underused (176 APs, ~0.7 clients, -65/-58 dBm); Hot (86 APs, ~3.2 clients, higher CPU/memory, -63/-58 dBm); Strong-signal mid-load (25 APs, ~2.2 clients, -59/-58 dBm).
  - Annealing k=3: Underused/weak-signal (117 APs, ~0.6 clients, -65/-61 dBm); Hot tier (88 APs, ~3.4 clients, higher CPU/memory, -63/-57 dBm); High-memory tier (82 APs, ~1 client, memory ~0.57, -63/-55 dBm).
- **Takeaways**
  - Baseline separates more cleanly (higher silhouette); annealing is tighter on SSE and highlights the high-memory tier.
  - “Silhouette” = how cleanly clusters separate (higher is better); “SSE” = how tight points are inside clusters (lower is better).
- **Limitations / future**
  - No interference/channel features; Euclidean k-means assumes spherical clusters.
  - Next: add RF/interference metrics, test multiple seeds/perturbations for stability, revisit k if actions need finer slices.
