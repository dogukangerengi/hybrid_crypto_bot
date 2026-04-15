# =============================================================================
# MODEL EVAL HARNESS — POST-DEPLOYMENT MONITORING
# =============================================================================
# Standalone script to re-run walk-forward evaluation against the live trade
# memory and produce a diagnostic report. Run after every retrain or weekly
# to detect performance degradation, regime shift, or data quality issues.
#
# Usage:
#   python -m ml.eval_harness --memory logs/ml_trade_memory.json --report logs/reports/diagnostic.xlsx
#
# Outputs:
#   1. Console summary with go/no-go signal
#   2. Excel report with per-fold metrics, feature stability, edge attribution
# =============================================================================

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .ensemble_model import EnsemblePredictor, MIN_TRAIN_SAMPLES

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# =============================================================================
# DECISION RULES — DEPLOYMENT GATES
# =============================================================================
# Production gates: model is approved for live trading only if ALL pass.
# Each gate is a falsifiable statistical claim, not heuristic intuition.

GATE_MIN_IC = 0.05               # Spearman IC floor (Grinold-Kahn actionable)
GATE_MIN_IR = 0.5                # Cross-fold stability (signal/noise ratio)
GATE_MIN_SPREAD = 0.20           # Long-short decile spread in R units
GATE_MAX_FOLD_DEGRADATION = 2    # Max consecutive folds with negative IC
GATE_MIN_AGGREGATED_Z = 1.65     # Aggregated Z (one-sided p ≤ 0.05)


def evaluate_edge_significance(returns: np.ndarray) -> dict:
    """
    Test whether the realized R-multiple distribution has positive expectancy.

    Uses one-sample t-test against H0: E[R] = 0, plus bootstrap CI for
    robustness against non-normality. Returns sufficient statistics for
    reporting.

    Failure mode: small n inflates variance estimate → conservative test.
    """
    n = len(returns)
    if n < 30:
        return {"n": n, "insufficient_data": True}

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    se_r = std_r / np.sqrt(n)
    t_stat, p_val = stats.ttest_1samp(returns, 0)

    # Bootstrap 95% CI (non-parametric, robust to fat tails)
    rng = np.random.RandomState(42)
    boot_means = np.array([
        np.mean(rng.choice(returns, size=n, replace=True))
        for _ in range(2000)
    ])
    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

    return {
        "n": n,
        "mean_r": mean_r,
        "std_r": std_r,
        "se_r": se_r,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "boot_ci_95": (float(ci_lower), float(ci_upper)),
        "per_trade_sharpe": mean_r / std_r if std_r > 0 else 0.0,
        "annualized_sharpe": (mean_r / std_r) * np.sqrt(250) if std_r > 0 else 0.0,
    }


def evaluate_directional_edge(df: pd.DataFrame) -> dict:
    """
    Decompose edge by trade direction (LONG vs SHORT).

    Tests whether either direction has statistically significant negative
    expectancy that warrants disabling. Asymmetric markets often produce
    asymmetric signal quality.
    """
    out = {}
    for d in df["direction"].unique():
        sub = df[df["direction"] == d]["r_multiple"].values
        if len(sub) < 30:
            out[d] = {"n": len(sub), "insufficient": True}
            continue
        t, p = stats.ttest_1samp(sub, 0)
        out[d] = {
            "n": len(sub),
            "mean_r": float(np.mean(sub)),
            "win_rate": float(np.mean(sub > 0)),
            "t_stat": float(t),
            "p_value": float(p),
            "negative_edge_significant": bool(t < 0 and p < 0.05),
        }
    return out


def evaluate_temporal_stability(df: pd.DataFrame, window: int = 50) -> dict:
    """
    Rolling-window edge stability check.

    Tests whether E[R] is stationary across the sample. Significant drift
    indicates regime change, model staleness, or strategy decay.
    """
    if len(df) < 2 * window:
        return {"insufficient_data": True}

    rolling_mean = df["r_multiple"].rolling(window).mean()
    rolling_mean = rolling_mean.dropna()

    # Mann-Kendall trend test (non-parametric)
    n = len(rolling_mean)
    s = sum(np.sign(rolling_mean.iloc[j] - rolling_mean.iloc[i])
            for i in range(n - 1) for j in range(i + 1, n))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var_s) if var_s > 0 else 0
    p_trend = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "rolling_mean_min": float(rolling_mean.min()),
        "rolling_mean_max": float(rolling_mean.max()),
        "rolling_mean_std": float(rolling_mean.std()),
        "mk_z_stat": float(z),
        "mk_p_value": float(p_trend),
        "trending_significantly": bool(p_trend < 0.05),
    }


def run_diagnostic(memory_path: str, report_path: str = None) -> dict:
    """
    Full diagnostic pipeline. Returns dict with all metrics and a
    binary deploy/no-deploy recommendation.
    """
    logger.info(f"Loading trade memory: {memory_path}")
    with open(memory_path) as f:
        trades = json.load(f)

    df = EnsemblePredictor.construct_r_multiple(trades)
    if df.empty:
        return {"status": "ERROR", "reason": "No closed trades in memory"}

    logger.info(f"Loaded {len(df)} closed trades with valid R-multiple")

    # 1. Edge significance test
    edge = evaluate_edge_significance(df["r_multiple"].values)
    logger.info(f"Edge: mean_R={edge.get('mean_r'):+.4f}, p={edge.get('p_value'):.4f}")

    # 2. Directional decomposition
    direction = evaluate_directional_edge(df)
    for d, stats_d in direction.items():
        if stats_d.get("negative_edge_significant"):
            logger.warning(f"  {d}: SIGNIFICANT NEGATIVE EDGE (p={stats_d['p_value']:.3f})")

    # 3. Temporal stability
    temporal = evaluate_temporal_stability(df)

    # 4. Walk-forward model evaluation
    feat_df = pd.json_normalize(df["features"]).reset_index(drop=True)
    y = df["r_multiple"].values
    directions = df["direction"].values   # YENİ: yön array'ini modele geçir

    logger.info("Running walk-forward model evaluation...")
    model = EnsemblePredictor()
    metrics = model.train(feat_df, y, directions=directions)   # GÜNCELLENDİ

    # 5. Aggregate Z under fold-IID assumption
    aggregated_z = (metrics.spearman_ic * np.sqrt(metrics.n_folds) /
                    metrics.ic_std) if metrics.ic_std > 1e-9 else 0.0

    # 6. Deployment gate evaluation
    gates = {
        "ic_above_floor": metrics.spearman_ic >= GATE_MIN_IC,
        "ir_above_floor": metrics.information_ratio >= GATE_MIN_IR,
        "spread_above_floor": metrics.long_short_spread >= GATE_MIN_SPREAD,
        "aggregated_z_significant": aggregated_z >= GATE_MIN_AGGREGATED_Z,
        "no_significant_negative_directional_edge": not any(
            d.get("negative_edge_significant") for d in direction.values()
        ),
    }
    deploy = all(gates.values())

    result = {
        "status": "EVALUATED",
        "n_trades": len(df),
        "edge_test": edge,
        "directional_edge": direction,
        "temporal_stability": temporal,
        "model_metrics": {
            "spearman_ic": metrics.spearman_ic,
            "ic_std": metrics.ic_std,
            "information_ratio": metrics.information_ratio,
            "mae": metrics.mae,
            "top_quintile_r": metrics.top_quintile_r,
            "bottom_quintile_r": metrics.bottom_quintile_r,
            "long_short_spread": metrics.long_short_spread,
            "n_folds": metrics.n_folds,
            "aggregated_z": aggregated_z,
        },
        "deployment_gates": gates,
        "deploy_recommendation": deploy,
    }

    if report_path:
        _write_excel_report(result, report_path)

    return result


def _write_excel_report(result: dict, path: str) -> None:
    """Write structured Excel report with three sheets."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([{
            "Metric": k, "Value": str(v)
        } for k, v in result["edge_test"].items()]).to_excel(
            writer, sheet_name="1_Edge_Test", index=False)

        rows = []
        for direction, stats_d in result["directional_edge"].items():
            row = {"direction": direction, **stats_d}
            rows.append(row)
        pd.DataFrame(rows).to_excel(writer, sheet_name="2_Directional", index=False)

        pd.DataFrame([{
            "Metric": k, "Value": str(v)
        } for k, v in result["model_metrics"].items()]).to_excel(
            writer, sheet_name="3_Model_Metrics", index=False)

        gates_rows = [
            {"gate": k, "passed": v}
            for k, v in result["deployment_gates"].items()
        ]
        gates_rows.append({"gate": "OVERALL_DEPLOY",
                           "passed": result["deploy_recommendation"]})
        pd.DataFrame(gates_rows).to_excel(writer, sheet_name="4_Deploy_Gates", index=False)


def print_summary(result: dict) -> None:
    """Console-friendly summary print."""
    print("\n" + "=" * 70)
    print("  MODEL DIAGNOSTIC REPORT")
    print("=" * 70)

    edge = result["edge_test"]
    print(f"\nSample Statistics:")
    print(f"  n trades       : {result['n_trades']}")
    print(f"  Mean R/trade   : {edge['mean_r']:+.4f}")
    print(f"  95% bootstrap CI: [{edge['boot_ci_95'][0]:+.4f}, {edge['boot_ci_95'][1]:+.4f}]")
    print(f"  Per-trade Sharpe: {edge['per_trade_sharpe']:+.3f}")

    print(f"\nDirectional Edge:")
    for d, stats_d in result["directional_edge"].items():
        if stats_d.get("insufficient"):
            print(f"  {d}: insufficient sample (n={stats_d['n']})")
            continue
        flag = " ⚠ NEG EDGE" if stats_d.get("negative_edge_significant") else ""
        print(f"  {d:5s}: n={stats_d['n']:3d} | mean R={stats_d['mean_r']:+.3f} | "
              f"WR={stats_d['win_rate']:.3f} | p={stats_d['p_value']:.3f}{flag}")

    m = result["model_metrics"]
    print(f"\nModel Metrics (Walk-Forward):")
    print(f"  Spearman IC      : {m['spearman_ic']:+.4f} ± {m['ic_std']:.4f}")
    print(f"  Information Ratio: {m['information_ratio']:+.2f}")
    print(f"  Long-short spread: {m['long_short_spread']:+.3f}R")
    print(f"  Aggregated Z     : {m['aggregated_z']:+.2f}")

    print(f"\nDeployment Gates:")
    for gate, passed in result["deployment_gates"].items():
        print(f"  [{'✓' if passed else '✗'}] {gate}")

    decision = "DEPLOY" if result["deploy_recommendation"] else "DO NOT DEPLOY"
    print(f"\n  RECOMMENDATION: {decision}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", required=True, help="Path to ml_trade_memory.json")
    parser.add_argument("--report", default=None, help="Optional Excel report path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    result = run_diagnostic(args.memory, args.report)
    print_summary(result)
