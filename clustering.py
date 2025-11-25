from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------------------
# Data loading and preparation
# ----------------------------

DATASET_DIR = Path("dataset") / "data 3 access point"


def load_raw_metrics() -> Dict[str, pd.DataFrame]:
    """Load the selected dataset files into DataFrames."""
    paths = {
        "client": DATASET_DIR / "client_metrics_uap_5min.csv.gz",
        "cpu": DATASET_DIR / "cpu_metrics_uap_5min.csv.gz",
        "mem": DATASET_DIR / "memory_metrics_uap_5min.csv.gz",
        "sig24": DATASET_DIR / "signal_24g_metrics_uap_5min.csv.gz",
        "sig5": DATASET_DIR / "signal_5g_metrics_uap_5min.csv.gz",
    }
    return {name: pd.read_csv(path) for name, path in paths.items()}


def aggregate_features(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate per-AP features; enforce complete rows."""
    client_agg = dfs["client"].groupby("ap_name")["client_count"].agg(["mean", "max", "std"])
    client_agg["std"] = client_agg["std"].fillna(0)

    cpu_agg = dfs["cpu"].groupby("ap_name")["cpu_usage_ratio"].agg(["mean", "max"])
    mem_agg = dfs["mem"].groupby("ap_name")["memory_usage_ratio"].agg(["mean", "max"])
    sig24_agg = dfs["sig24"].groupby("ap_name")["signal_dbm"].agg(["mean", "min", "max"])
    sig5_agg = dfs["sig5"].groupby("ap_name")["signal_dbm"].agg(["mean", "min", "max"])

    features = pd.concat(
        [
            client_agg.add_prefix("clients_"),
            cpu_agg.add_prefix("cpu_"),
            mem_agg.add_prefix("mem_"),
            sig24_agg.add_prefix("sig24_"),
            sig5_agg.add_prefix("sig5_"),
        ],
        axis=1,
    )

    # Keep all APs: impute any missing metric with the feature's mean instead of dropping rows.
    features = features.apply(lambda col: col.fillna(col.mean()))

    return features


def standardize(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardization: returns (scaled, mean, std)."""
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    return (values - mean) / std, mean, std


# ----------------------------
# Metrics
# ----------------------------

def sse_and_labels(data: np.ndarray, centroids: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute SSE and hard assignments for given centroids."""
    d2 = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    sse = float(d2[np.arange(len(data)), labels].sum())
    return sse, labels


def silhouette_values(data: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-sample silhouette values."""
    unique = np.unique(labels)
    if len(unique) == 1:
        return np.zeros(len(data))

    pairwise = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
    silhouettes = np.zeros(len(data))
    for i in range(len(data)):
        same = labels == labels[i]
        if same.sum() > 1:
            a = pairwise[i, same].sum() / (same.sum() - 1)  # exclude self
        else:
            a = 0.0
        b = min(pairwise[i, labels == c].mean() for c in unique if c != labels[i])
        silhouettes[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return silhouettes


def manual_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """Manual silhouette without external libraries (mean of per-sample values)."""
    return float(silhouette_values(data, labels).mean())


def davies_bouldin_index(data: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin index: lower is better."""
    unique = np.unique(labels)
    k = len(unique)
    if k < 2:
        return float("nan")
    centroids = np.array([data[labels == c].mean(axis=0) for c in unique])
    scatter = np.array(
        [np.mean(np.linalg.norm(data[labels == c] - centroids[i], axis=1)) if (labels == c).any() else 0.0 for i, c in enumerate(unique)]
    )
    dbi = 0.0
    for i in range(k):
        ratios = []
        for j in range(k):
            if i == j:
                continue
            dist = np.linalg.norm(centroids[i] - centroids[j])
            ratios.append((scatter[i] + scatter[j]) / max(dist, 1e-12))
        dbi += max(ratios)
    return float(dbi / k)


def calinski_harabasz_index(data: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz index: higher is better."""
    unique = np.unique(labels)
    k = len(unique)
    n = len(data)
    if k < 2 or n == k:
        return float("nan")
    overall_mean = data.mean(axis=0)
    centroids = np.array([data[labels == c].mean(axis=0) for c in unique])
    sizes = np.array([(labels == c).sum() for c in unique])
    between = sum(sizes[i] * np.linalg.norm(centroids[i] - overall_mean) ** 2 for i in range(k))
    within = sum(np.sum((data[labels == c] - centroids[i]) ** 2) for i, c in enumerate(unique))
    return float((between / max(k - 1, 1e-12)) / (within / max(n - k, 1e-12)))


# ----------------------------
# Baseline clustering: k-means SSE + multi-start k-means++
# ----------------------------

def kmeans_pp_init(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding."""
    n = len(data)
    centroids = [data[rng.integers(0, n)]]
    while len(centroids) < k:
        d2 = np.min(((data[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(axis=2), axis=1)
        probs = d2 / d2.sum()
        centroids.append(data[rng.choice(n, p=probs)])
    return np.array(centroids)


def lloyd_kmeans(
    data: np.ndarray, centroids: np.ndarray, max_iter: int = 100, tol: float = 1e-4, rng: np.random.Generator | None = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Standard Lloyd's updates for k-means SSE."""
    rng = rng or np.random.default_rng()
    k = centroids.shape[0]
    for _ in range(max_iter):
        sse, labels = sse_and_labels(data, centroids)
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centroids[j] = data[mask].mean(axis=0)
            else:
                new_centroids[j] = data[rng.integers(0, len(data))]
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break
    sse, labels = sse_and_labels(data, centroids)
    return centroids, labels, sse


def multi_start_kmeans(
    data: np.ndarray,
    k_range: Iterable[int],
    restarts: int = 100,
    min_cluster_size: int = 5,
    seed: int = 99,
) -> Dict[str, object]:
    """
    Explore k over k_range with many restarts; filter out tiny clusters; pick by highest silhouette then lowest SSE.
    Returns a dict with centroids, labels, metrics.
    """
    rng = np.random.default_rng(seed)
    best = None
    for k in k_range:
        for _ in range(restarts):
            centroids = kmeans_pp_init(data, k, rng)
            centroids, labels, sse = lloyd_kmeans(data, centroids, rng=rng)
            sizes = np.bincount(labels, minlength=k)
            if sizes.min() < min_cluster_size:
                continue
            sil = manual_silhouette(data, labels)
            key = (sil, -sse)
            if (best is None) or (key > best["key"]):
                best = {
                    "key": key,
                    "k": k,
                    "centroids": centroids.copy(),
                    "labels": labels.copy(),
                    "sizes": sizes,
                    "sse": sse,
                    "silhouette": sil,
                }
    if best is None:
        raise RuntimeError("No valid clustering found; try relaxing min_cluster_size or adjusting k_range.")
    return best


# ----------------------------
# Optimizer: Deterministic Annealing Clustering (DAC)
# ----------------------------

def deterministic_annealing_cluster(
    data: np.ndarray,
    k: int = 3,
    T0: float = 6.0,
    Tmin: float = 0.01,
    alpha: float = 0.9,
    inner_steps: int = 12,
    seed: int = 123,
) -> Dict[str, object]:
    """
    Deterministic annealing for k-means SSE.
    Temperature controls assignment softness; cooling tightens assignments and refines centroids.
    """
    rng = np.random.default_rng(seed)
    n, _ = data.shape
    centroids = data[rng.choice(n, k, replace=False)].copy()
    best_centroids = centroids.copy()

    def current_sse_labels(cents: np.ndarray) -> Tuple[float, np.ndarray]:
        d2 = ((data[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        sse = float(d2[np.arange(n), labels].sum())
        return sse, labels

    sse, labels = current_sse_labels(centroids)
    best_sse = sse
    best_labels = labels.copy()
    history: List[Tuple[float, float]] = []

    T = T0
    while T > Tmin:
        for _ in range(inner_steps):
            d2 = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            weights = np.exp(-d2 / max(T, 1e-8))
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)
            denom = weights.sum(axis=0)[:, None] + 1e-12
            centroids = (weights.T @ data) / denom
        sse, labels = current_sse_labels(centroids)
        history.append((T, sse))
        if sse < best_sse:
            best_sse = sse
            best_centroids = centroids.copy()
            best_labels = labels.copy()
        T *= alpha

    final_sse, final_labels = current_sse_labels(best_centroids)
    final_silhouette = manual_silhouette(data, final_labels)
    return {
        "centroids": best_centroids,
        "labels": final_labels,
        "sse": final_sse,
        "silhouette": final_silhouette,
        "sizes": np.bincount(final_labels, minlength=k),
        "history": history,
        "k": k,
    }


# ----------------------------
# Reporting helpers
# ----------------------------

def summarize_clusters(features: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, object]]:
    """Compute human-friendly summaries for each cluster."""
    summaries = []
    for c in np.unique(labels):
        mask = labels == c
        cluster_df = features.iloc[mask]
        summaries.append(
            {
                "cluster": int(c),
                "size": int(mask.sum()),
                "clients_mean": cluster_df["clients_mean"].mean(),
                "clients_max": cluster_df["clients_max"].mean(),
                "cpu_mean": cluster_df["cpu_mean"].mean(),
                "mem_mean": cluster_df["mem_mean"].mean(),
                "sig24_mean": cluster_df["sig24_mean"].mean(),
                "sig5_mean": cluster_df["sig5_mean"].mean(),
            }
        )
    return sorted(summaries, key=lambda x: -x["size"])


def silhouette_summary(values: np.ndarray) -> Dict[str, float]:
    """Five-number summary for silhouette values."""
    return {
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def save_results(results: Dict[str, object]) -> None:
    """Persist results to JSON and TXT for the report."""
    with open("results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "Clustering results",
        f"Baseline MS k-means++ (k={results['baseline']['k']}): SSE={results['baseline']['sse']:.2f}, "
        f"silhouette={results['baseline']['silhouette']:.3f}, sizes={results['baseline']['sizes']}, "
        f"DBI={results['baseline']['dbi']:.3f}, CH={results['baseline']['calinski_harabasz']:.2f}",
        f"Deterministic Annealing (k={results['annealing']['k']}): SSE={results['annealing']['sse']:.2f}, "
        f"silhouette={results['annealing']['silhouette']:.3f}, sizes={results['annealing']['sizes']}, "
        f"DBI={results['annealing']['dbi']:.3f}, CH={results['annealing']['calinski_harabasz']:.2f}",
        "",
        "Silhouette summary (baseline):",
        json.dumps(results["baseline"]["silhouette_summary"]),
        "Silhouette summary (annealing):",
        json.dumps(results["annealing"]["silhouette_summary"]),
        "",
        "k-scan (baseline):",
        pd.DataFrame(results["baseline_scan"]).to_csv(index=False),
        "k-scan (annealing):",
        pd.DataFrame(results["annealing_scan"]).to_csv(index=False),
        "",
        "Baseline summary (per cluster):",
        pd.DataFrame(results["baseline"]["summary"]).to_csv(index=False),
        "Annealing summary (per cluster):",
        pd.DataFrame(results["annealing"]["summary"]).to_csv(index=False),
    ]

    with open("results_summary.txt", "w") as f:
        f.write("\n".join(lines))


# ----------------------------
# Visualization helpers
# ----------------------------

def pca_project(data: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple PCA via SVD; returns projected data, components, and mean."""
    mean = data.mean(axis=0)
    centered = data - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].T
    projected = centered @ components
    return projected, components, mean


def project_points(points: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Project additional points using precomputed PCA components."""
    return (points - mean) @ components


def plot_clusters(
    coords: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    title: str,
    outfile: str,
) -> None:
    """Scatter plot of clusters and centroids in 2D."""
    plt.figure(figsize=(7, 5))
    palette = plt.cm.tab10.colors
    unique = np.unique(labels)
    for c in unique:
        mask = labels == c
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=25,
            color=palette[c % len(palette)],
            alpha=0.7,
            label=f"Cluster {c} (n={mask.sum()})",
        )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=140,
        marker="X",
        color="black",
        label="Centroids",
        edgecolor="white",
        linewidth=1.0,
    )
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_silhouette_hist(baseline_vals: np.ndarray, anneal_vals: np.ndarray, outfile: str) -> None:
    """Side-by-side silhouette histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    axes[0].hist(baseline_vals, bins=20, color="#3b82f6", alpha=0.8, edgecolor="white")
    axes[0].set_title("Baseline silhouettes")
    axes[0].set_xlabel("Silhouette")
    axes[0].set_ylabel("Count")

    axes[1].hist(anneal_vals, bins=20, color="#10b981", alpha=0.8, edgecolor="white")
    axes[1].set_title("Annealing silhouettes")
    axes[1].set_xlabel("Silhouette")

    fig.suptitle("Silhouette distribution by method", fontsize=12)
    fig.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_cooling_curve(history: List[Tuple[float, float]], outfile: str) -> None:
    """Plot SSE across the annealing temperature schedule."""
    if not history:
        return
    temps, sses = zip(*history)
    plt.figure(figsize=(6, 4))
    plt.plot(temps, sses, marker="o", color="#f59e0b")
    plt.gca().invert_xaxis()
    plt.xlabel("Temperature (T)")
    plt.ylabel("SSE")
    plt.title("Deterministic annealing cooling path")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def plot_k_scan(
    k_values: List[int],
    baseline_metrics: List[Dict[str, float]],
    anneal_metrics: List[Dict[str, float]],
    outfile: str,
) -> None:
    """Line plots of SSE and silhouette over k for both methods."""
    plt.figure(figsize=(8, 5))
    baseline_sse = [m["sse"] for m in baseline_metrics]
    anneal_sse = [m["sse"] for m in anneal_metrics]
    baseline_sil = [m["silhouette"] for m in baseline_metrics]
    anneal_sil = [m["silhouette"] for m in anneal_metrics]

    ax1 = plt.gca()
    ax1.plot(k_values, baseline_sse, label="Baseline SSE", marker="o", color="#2563eb")
    ax1.plot(k_values, anneal_sse, label="Annealing SSE", marker="o", color="#f97316")
    ax1.set_xlabel("k")
    ax1.set_ylabel("SSE")
    ax1.tick_params(axis="y", labelcolor="black")

    ax2 = ax1.twinx()
    ax2.plot(k_values, baseline_sil, label="Baseline silhouette", marker="s", linestyle="--", color="#1d4ed8")
    ax2.plot(k_values, anneal_sil, label="Annealing silhouette", marker="s", linestyle="--", color="#ea580c")
    ax2.set_ylabel("Silhouette")
    ax2.tick_params(axis="y", labelcolor="black")

    lines_labels = ax1.get_legend_handles_labels()
    lines_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_labels[0] + lines_labels2[0], lines_labels[1] + lines_labels2[1], loc="best")
    plt.title("Model quality vs. k")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


# ----------------------------
# Main pipeline
# ----------------------------

def main() -> None:
    # 1) Load and aggregate features
    dfs = load_raw_metrics()
    features = aggregate_features(dfs)
    print(f"Loaded features: {features.shape[0]} APs x {features.shape[1]} features")

    # 2) Standardize
    X = features.values.astype(float)
    X_scaled, mean, std = standardize(X)

    # 3) Baseline: multi-start k-means++ fixed at k=3
    baseline = multi_start_kmeans(X_scaled, k_range=[3], restarts=120, min_cluster_size=5, seed=99)
    baseline_summary = summarize_clusters(features, baseline["labels"])
    print(
        f"Baseline MS k-means++ (k=3): SSE={baseline['sse']:.2f}, "
        f"silhouette={baseline['silhouette']:.3f}, sizes={baseline['sizes'].tolist()}"
    )

    # 4) Optimizer: deterministic annealing fixed at k=3 (niche, course-relevant)
    annealing = deterministic_annealing_cluster(X_scaled, k=3, T0=6.0, Tmin=0.01, alpha=0.9, inner_steps=12, seed=123)
    annealing_summary = summarize_clusters(features, annealing["labels"])
    print(
        f"Deterministic annealing (k=3): SSE={annealing['sse']:.2f}, "
        f"silhouette={annealing['silhouette']:.3f}, sizes={annealing['sizes'].tolist()}"
    )

    # 5) Package results
    results = {
        "dataset": "data 3 access point",
        "n_samples": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "standardization": {"mean": mean.tolist(), "std": std.tolist()},
        "baseline": {
            "method": "multi-start k-means++ + Lloyd",
            "k": int(baseline["k"]),
            "sse": float(baseline["sse"]),
            "silhouette": float(baseline["silhouette"]),
            "sizes": baseline["sizes"].tolist(),
            "summary": baseline_summary,
        },
        "annealing": {
            "method": "deterministic annealing for k-means SSE",
            "k": int(annealing["k"]),
            "sse": float(annealing["sse"]),
            "silhouette": float(annealing["silhouette"]),
            "sizes": annealing["sizes"].tolist(),
            "summary": annealing_summary,
            "history": annealing["history"],
        },
        "best_method": "baseline" if baseline["silhouette"] >= annealing["silhouette"] else "annealing",
    }

    # 6) Visualize clusters in 2D (PCA projection)
    coords, comps, mean_pca = pca_project(X_scaled, n_components=2)
    baseline_centroids_proj = project_points(baseline["centroids"], comps, mean_pca)
    annealing_centroids_proj = project_points(annealing["centroids"], comps, mean_pca)
    plot_clusters(coords, baseline["labels"], baseline_centroids_proj, "Baseline k-means++ (PCA 2D)", "baseline_clusters.png")
    plot_clusters(coords, annealing["labels"], annealing_centroids_proj, "Deterministic Annealing (PCA 2D)", "annealing_clusters.png")
    print("Saved PCA visualizations to baseline_clusters.png and annealing_clusters.png")

    # 7) Save
    save_results(results)
    print("Wrote results_summary.json and results_summary.txt")


if __name__ == "__main__":
    main()
