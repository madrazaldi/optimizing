# Presentation Script (slide-by-slide)

**Slide 1 – Title**  
“Hi, I’m Muhammad Razan Alamudi. This is ‘Wi-Fi AP Clustering with Deterministic Annealing’—dataset ‘data 3 access point,’ 287 APs, pandas + numpy only.”

**Slide 2 – Problem & Goal (why ops should care)**  
“Ops teams need to know which APs are hot, steady, or underused. The goal is simple buckets so we can relieve, monitor, or repurpose gear. Constraint: no ML libraries, so we optimized a basic k-means objective with a niche method.”

**Slide 3 – Data & Features**  
“We pulled five gzipped CSVs: clients, CPU, memory, and signal on 2.4G and 5G. That yields 13 features per AP—load stats, CPU/memory, and signal min/mean/max on both bands. Any client-std NaNs (constant series) were set to 0. After imputation we have 287 APs and standardized features.”

**Slide 4 – Preprocessing**  
“We filled missing metrics with feature means so no AP is dropped. Then we z-scored everything so no single metric dominates. Integrity checks: shape is 287×13, no NaNs remain, and `ap_name` is preserved for interpretation.”

**Slide 5 – Methods Compared (plain English)**  
“Two grouping methods on the same objective:  

1) Baseline: run k-means++ many times at k=3, enforce at least 5 APs per cluster, keep the split with the best separation.  
2) Optimizer: deterministic annealing—start with fuzzy assignments that harden as we cool; fixed k=3; keep the tightest split. Seeds: 99 and 123.”

**Slide 6 – How we judged the splits**  
“Two scores: SSE (tightness inside clusters—lower is better) and silhouette (separation between clusters—higher is better). Results: Baseline k=3 → SSE 2803.64, silhouette 0.212, sizes [86, 25, 176]. Annealing k=3 → SSE 2731.39, silhouette 0.166, sizes [88, 82, 117]. Trade-off: baseline separates better; annealing is tighter but more overlapped.”

**Slide 7 – Baseline Cluster Narratives (k=3)**  
“Three clusters:  
- Hot (86): ~3.2 clients, higher CPU/memory, signals ~ -63/-58 dBm → relieve/strengthen.  
- Strong-signal mid-load (25): ~2.2 clients, balanced CPU/memory, signals ~ -59/-58 dBm → monitor/right-size.  
- Underused (176): ~0.7 clients, signals ~ -65/-58 dBm → repurpose/retune.”

**Slide 8 – Annealing Cluster Narratives (k=3)**  
“Three tiers:  
- Hot (88): ~3.4 clients, higher CPU/memory, signals ~ -63/-57 dBm → relieve/strengthen.  
- High-memory (82): ~1 client, memory ~0.57, signals ~ -63/-55 dBm → audit/optimize.  
- Underused/weak-signal (117): ~0.6 clients, signals ~ -65/-61 dBm → repurpose/retune.”

**Slide 9 – Visuals (what the plots show)**  
“Here are the PCA 2D plots. Each dot is an AP projected to 2D; colors are clusters; X marks centroids. Left: baseline k=3 (`baseline_clusters.png`, title: ‘Baseline k=3 (PCA 2D)’). Right: annealing k=3 (`annealing_clusters.png`, title: ‘Deterministic Annealing k=3 (PCA 2D)’). Baseline separates a bit more cleanly; annealing compacts clusters more.”

**Slide 10 – Recommended action path**  
“Use k=3 labels for the playbook: relieve hot, audit high-memory/strong-signal, repurpose/retune underused. Choose baseline labels for clearer separation; choose annealing labels for tighter SSE and a pronounced high-memory tier.”

**Slide 11 – Limitations & Next Steps**  
“Limits: no interference/channel metrics; k-means assumes spherical clusters; silhouettes softened because we kept all APs via imputation. Next: add RF/interference features, test more seeds/perturbations, revisit k, and try non-spherical methods.”

**Slide 12 – Backup / Appendix**  
“Hyperparameters for the curious: annealing T0=6.0, alpha=0.9, inner steps=12; baseline restarts=120, min cluster size=5. Also have SSE history for annealing and a note that dropping rows vs. imputing changes counts and scores.” 
