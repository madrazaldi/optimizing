from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
    ).dropna()

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


def manual_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    """Manual silhouette without external libraries."""
    unique = np.unique(labels)
    if len(unique) == 1:
        return 0.0
    pairwise = np.linalg.norm(data[:, None, :] - data[None, :, :], axis=2)
    silhouettes = []
    for i in range(len(data)):
        same = labels == labels[i]
        a = pairwise[i, same].mean() if same.sum() > 1 else 0
        b = min(pairwise[i, labels == c].mean() for c in unique if c != labels[i])
        silhouettes.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
    return float(np.mean(silhouettes))


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


def save_results(results: Dict[str, object]) -> None:
    """Persist results to JSON and TXT for the report."""
    with open("results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "Clustering results",
        f"Baseline MS k-means++ (k={results['baseline']['k']}): SSE={results['baseline']['sse']:.2f}, "
        f"silhouette={results['baseline']['silhouette']:.3f}, sizes={results['baseline']['sizes']}",
        f"Deterministic Annealing (k={results['annealing']['k']}): SSE={results['annealing']['sse']:.2f}, "
        f"silhouette={results['annealing']['silhouette']:.3f}, sizes={results['annealing']['sizes']}",
        "",
        "Baseline summary (per cluster):",
        pd.DataFrame(results["baseline"]["summary"]).to_csv(index=False),
        "Annealing summary (per cluster):",
        pd.DataFrame(results["annealing"]["summary"]).to_csv(index=False),
    ]
    with open("results_summary.txt", "w") as f:
        f.write("\n".join(lines))


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

    # 3) Baseline: multi-start k-means++ across k=2..6
    baseline = multi_start_kmeans(X_scaled, k_range=range(2, 7), restarts=120, min_cluster_size=5, seed=99)
    baseline_summary = summarize_clusters(features, baseline["labels"])
    print(
        f"Baseline MS k-means++: k={baseline['k']}, SSE={baseline['sse']:.2f}, "
        f"silhouette={baseline['silhouette']:.3f}, sizes={baseline['sizes'].tolist()}"
    )

    # 4) Optimizer: deterministic annealing for k=3 (niche, course-relevant)
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

    # 6) Save
    save_results(results)
    print("Wrote results_summary.json and results_summary.txt")


if __name__ == "__main__":
    main()
