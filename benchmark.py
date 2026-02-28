import csv
from datetime import datetime
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


# -----------------------------
# GHZ4 con folding (H e CX auto-inversi)
# -----------------------------
def fold_gate(qc, gate_name, qubits, f):
    for _ in range(f):
        if gate_name == "h":
            qc.h(qubits[0])
        elif gate_name == "cx":
            qc.cx(qubits[0], qubits[1])


def ghz4_folded(fold_factor=1):
    qc = QuantumCircuit(4, 4)
    fold_gate(qc, "h",  [0], fold_factor)
    fold_gate(qc, "cx", [0, 1], fold_factor)
    fold_gate(qc, "cx", [1, 2], fold_factor)
    fold_gate(qc, "cx", [2, 3], fold_factor)
    return qc


def measure_Z(qc):
    qc = qc.copy()
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    return qc


def measure_X(qc):
    qc = qc.copy()
    qc.h(0); qc.h(1); qc.h(2); qc.h(3)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    return qc


def parity_expectation(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    acc = 0.0
    for bitstring, c in counts.items():
        parity = (-1) ** (bitstring.count("1"))
        acc += parity * c
    return acc / total


def run_obs(noise_p, fold_factor, basis="Z", shots=1500, seed=1):
    base = ghz4_folded(fold_factor)
    qc = measure_Z(base) if basis == "Z" else measure_X(base)

    # noise model: depolarizing su H e CX
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(noise_p, 1), ['h'])

    # per gate 2-qubit, amplifico un po' (come prima)
    nm.add_all_qubit_quantum_error(depolarizing_error(min(2 * noise_p, 0.95), 2), ['cx'])

    sim = AerSimulator(noise_model=nm, seed_simulator=seed)
    qc_t = transpile(qc, sim, seed_transpiler=seed)
    res = sim.run(qc_t, shots=shots).result()
    counts = res.get_counts()
    return parity_expectation(counts)


# -----------------------------
# ZNE fit + metriche
# -----------------------------
def fit_and_score(x, y, degree):
    coeff = np.polyfit(x, y, degree)
    y_hat = np.polyval(coeff, x)
    resid = y - y_hat
    sse = float(np.sum(resid ** 2))

    k = degree + 1
    n = len(x)
    aic = n * np.log(sse / n + 1e-12) + 2 * k

    zne0 = float(np.polyval(coeff, 0.0))
    return zne0, sse, aic


def ci95(arr):
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_bootstrap(tag, z0_lin, z0_quad, aic_lin, aic_quad):
    z0_lin = np.array(z0_lin); z0_quad = np.array(z0_quad)
    aic_lin = np.array(aic_lin); aic_quad = np.array(aic_quad)

    mean_lin = float(z0_lin.mean())
    mean_quad = float(z0_quad.mean())
    std_lin = float(z0_lin.std(ddof=1))
    std_quad = float(z0_quad.std(ddof=1))
    ci_lin = ci95(z0_lin)
    ci_quad = ci95(z0_quad)

    mean_aic_lin = float(aic_lin.mean())
    mean_aic_quad = float(aic_quad.mean())
    winner = "Lineare" if mean_aic_lin < mean_aic_quad else "Quadratico"

    if winner == "Lineare":
        zne_mean, zne_std, (lo, hi) = mean_lin, std_lin, ci_lin
        best_aic = mean_aic_lin
    else:
        zne_mean, zne_std, (lo, hi) = mean_quad, std_quad, ci_quad
        best_aic = mean_aic_quad

    return {
        "tag": tag,
        "winner": winner,
        "zne_mean": zne_mean,
        "zne_std": zne_std,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_width": hi - lo,
        "aic_lin": mean_aic_lin,
        "aic_quad": mean_aic_quad,
        "best_aic": best_aic,
    }


# -----------------------------
# Breakdown rules
# -----------------------------
def breakdown_flags(summary, ci_width_threshold=0.35):
    out_of_range = (summary["zne_mean"] < -1.0) or (summary["zne_mean"] > 1.0)
    wide_ci = summary["ci_width"] > ci_width_threshold
    ci_outside = (summary["ci_lo"] < -1.0) or (summary["ci_hi"] > 1.0)
    return out_of_range, wide_ci, ci_outside


def main():
    experiment_id = "GHZ4_ZNE_BREAKDOWN_MAP_FOLDING_EXTENDED"

    folds = np.array([1, 3, 5, 7, 9, 11], dtype=float)
    noise_grid = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    shots = 1500
    B = 15  # bootstrap (non troppo pesante)

    # se CI più largo di 0.35 => instabile
    ci_width_threshold = 0.35

    fname = f"exp_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print("=== MAPPA BREAKDOWN ZNE (folding) EXTENDED ===")
    print("folds =", list(folds.astype(int)), "| noise_grid =", noise_grid)
    print("shots =", shots, "| B =", B)
    print("CSV:", fname)

    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "experiment_id", "noise_p", "basis", "folds",
            "B", "shots",
            "model_selected",
            "zne_mean", "zne_std", "ci_lo", "ci_hi", "ci_width",
            "AIC_lineare", "AIC_quadratico",
            "flag_out_of_range", "flag_wide_CI", "flag_CI_outside_range"
        ])

        for noise_p in noise_grid:
            print(f"\n--- noise_p = {noise_p} ---")

            for basis in ["Z", "X"]:
                z0_lin, z0_quad = [], []
                aic_lin, aic_quad = [], []

                for b in range(B):
                    seed = 12000 + b + int(noise_p * 1000)

                    y = []
                    for ff in folds:
                        val = run_obs(noise_p, int(ff), basis=basis, shots=shots, seed=seed)
                        y.append(val)
                    y = np.array(y, dtype=float)

                    z_l, sse_l, a_l = fit_and_score(folds, y, degree=1)
                    z_q, sse_q, a_q = fit_and_score(folds, y, degree=2)

                    z0_lin.append(z_l); aic_lin.append(a_l)
                    z0_quad.append(z_q); aic_quad.append(a_q)

                tag = "<ZZZZ>" if basis == "Z" else "<XXXX>"
                summary = summarize_bootstrap(tag, z0_lin, z0_quad, aic_lin, aic_quad)
                out_of_range, wide_ci, ci_outside = breakdown_flags(summary, ci_width_threshold)

                print(f"{tag} modello={summary['winner']:<10} "
                      f"ZNE0={summary['zne_mean']:.3f}  "
                      f"CI95=({summary['ci_lo']:.3f},{summary['ci_hi']:.3f})  "
                      f"width={summary['ci_width']:.3f}  "
                      f"flags: out={out_of_range} wideCI={wide_ci}")

                w.writerow([
                    experiment_id, noise_p, basis, ";".join(map(str, folds.astype(int))),
                    B, shots,
                    summary["winner"],
                    summary["zne_mean"], summary["zne_std"], summary["ci_lo"], summary["ci_hi"], summary["ci_width"],
                    summary["aic_lin"], summary["aic_quad"],
                    out_of_range, wide_ci, ci_outside
                ])

    print("\n=== FINITO ===")
    print("CSV salvato:", fname)
    print("Se vedi out=True o wideCI=True, sei nella zona dove ZNE rompe/instabile.")
    input("\nPremi invio per chiudere...")


if __name__ == "__main__":
    main()