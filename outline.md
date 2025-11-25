# Presentation Outline (Deterministic Annealing vs. K-means++)

- **Slide 1 – Title**
  - Title: “Wi-Fi AP Clustering with Deterministic Annealing”
  - Subtitle: dataset “data 3 access point” | 287 APs | pandas + numpy only
  - Names, course, date

- **Slide 2 – Problem & Goal (why ops should care)**
  - Pain: unclear which APs are hot, steady, or underused
  - Goal: simple buckets so ops can relieve, monitor, or repurpose APs
  - Constraint: no ML libs; optimize k-means SSE with a niche method

- **Slide 3 – Data & Features**
  - Source: 5 gzipped CSVs (clients, CPU, memory, 2.4G/5G signal)
  - 13 features per AP (clients mean/max/std; CPU mean/max; memory mean/max; signal mean/min/max on 2.4G & 5G)
  - After mean imputation: 287 APs × 13 features
  - Note: client std NaNs set to 0 for constant series

- **Slide 4 – Preprocessing**
  - Mean-impute missing metrics per feature (keeps all APs)
  - Z-score scale features
  - Integrity checks: shape, no NaNs, preserved `ap_name`

- **Slide 5 – Methods Compared (plain English)**
  - Baseline: run k-means++ with many restarts at k=3; enforce at least 5 APs per cluster; keep the best silhouette.
  - Optimizer: deterministic annealing—start fuzzy, cool to hard groups; fixed k=3; track best SSE.
  - Seeds: 99 (baseline), 123 (annealing)

- **Extra Slide**: Baseline Flowchart

```
flowchart TD
  A[Start: pick dataset (data 3 AP)] --> B[Load 5 CSVs (clients, CPU, memory, 2.4G, 5G)]
  B --> C[Aggregate per-AP stats (13 features)]
  C --> D[Impute feature means; keep all 287 APs]
  D --> E[Z-score standardize features]
  E --> F{Restart loop: 120 runs}
  F --> G[k-means++ init (k=3, seed=99)]
  G --> H[Lloyd updates until shift <1e-4 or 100 iters]
  H --> I{Any cluster size <5?}
  I -- yes --> F
  I -- no --> J[Compute SSE + silhouette]
  J --> K{Best silhouette? tie→lowest SSE}
  K --> F
  F -->|after all restarts| L[Keep best centroids/labels]
  L --> M[Report SSE, silhouette, sizes, cluster profiles]
  M --> N[End]
```

- **Extra Slide**: Annealing Flowchart

```
flowchart TD
  A[Start: same dataset + preprocessing] --> B[Aggregate → Impute → Standardize]
  B --> C{Path}
  C --> D[Baseline: multi-start k-means++ (k=3, 120 restarts)]
  C --> E[Optimizer: deterministic annealing (k=3)]
  D --> F[Baseline output: best silhouette, SSE, sizes]
  E --> G[Init random centroids (seed=123)]
  G --> H[Set T=6.0; Tmin=0.01; alpha=0.9; 12 refinements/level]
  H --> I{While T >= Tmin}
  I --> J[Compute soft weights exp(-d^2/T)]
  J --> K[Update centroids with weighted means]
  K --> L[Track best SSE/centroids along cooling path]
  L --> M[Lower T ← alpha·T]
  M --> I
  I -->|done| N[Hard-assign to nearest centroid; compute SSE + silhouette]
  N --> O[Annealing output: SSE, silhouette, sizes, profiles]
  F --> P[Compare SSE/silhouette/cluster shapes]
  O --> P
  P --> Q[Select/interpret per objective (tightness vs separation)]
  Q --> R[End]
```

- **Slide 6 – How we judged the splits**
  - SSE: tightness inside clusters (lower is better)
  - Silhouette: separation between clusters (higher is better)
  - Baseline (k=3): SSE 2803.64, silhouette 0.212, sizes [86, 25, 176]
  - Annealing (k=3): SSE 2731.39, silhouette 0.166, sizes [88, 82, 117]
  - Trade-off: baseline separates better (higher silhouette); annealing is tighter (lower SSE) but more overlap.

- **Slide 7 – Baseline Cluster Narratives (k=3)**
  - Hot (86): ~3.2 clients, higher CPU/memory, -63/-58 dBm → relieve/strengthen
  - Strong-signal mid-load (25): ~2.2 clients, balanced CPU/memory, -59/-58 dBm → monitor, ensure capacity is right-sized
  - Underused (176): ~0.7 clients, -65/-58 dBm → repurpose/retune

- **Slide 8 – Annealing Cluster Narratives (k=3)**
  - Hot (88): ~3.4 clients, higher CPU/memory, -63/-57 dBm → relieve/strengthen
  - High-memory (82): ~1.0 clients, memory ~0.57, -63/-55 dBm → audit/optimize
  - Underused/weak-signal (117): ~0.6 clients, -65/-61 dBm → repurpose/retune

- **Slide 9 – Visuals (what the plots show)**
  - Insert `annealing_clusters.png` (title: “Deterministic Annealing k=3 (PCA 2D)”)
  - Insert `baseline_clusters.png` (title: “Baseline k=3 (PCA 2D)”)
  - Explain: each dot = an AP projected to 2D; colors = clusters; X = centroids. Baseline shows cleaner separation; annealing is tighter but overlaps more in 2D projection.

