import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

MILP_DIR = Path("./output")

OBJECTIVES = ("latency", "energy", "throughput")
MINIMIZE = [True, True, False]  # latency ↓, energy ↓, throughput ↑

OUT_PARETO = Path("pareto_milp_fronts.png")

# ------------------------------------------------------------
# Pareto dominance
# ------------------------------------------------------------

def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    better_or_equal = (
        a["latency"] <= b["latency"] and
        a["energy"] <= b["energy"] and
        a["throughput"] >= b["throughput"]
    )

    strictly_better = (
        a["latency"] < b["latency"] or
        a["energy"] < b["energy"] or
        a["throughput"] > b["throughput"]
    )

    return better_or_equal and strictly_better


# ------------------------------------------------------------
# Fast non-dominated sorting
# ------------------------------------------------------------

def fast_nondominated_sort(
    population: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    S = {}
    n = {}
    fronts: list[list[dict[str, Any]]] = [[]]

    for p in population:
        pid = p["id"]
        S[pid] = []
        n[pid] = 0

        for q in population:
            if pid == q["id"]:
                continue

            if dominates(p["objectives"], q["objectives"]):
                S[pid].append(q)
            elif dominates(q["objectives"], p["objectives"]):
                n[pid] += 1

        if n[pid] == 0:
            p["rank"] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p["id"]]:
                n[q["id"]] -= 1
                if n[q["id"]] == 0:
                    q["rank"] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


# ------------------------------------------------------------
# Load MILP population
# ------------------------------------------------------------

def load_milp_population(base_dir: Path) -> list[dict]:
    population = []

    for case_dir in sorted(base_dir.iterdir()):
        obj_file = case_dir / "objectives.json"
        if not obj_file.exists():
            continue

        with open(obj_file, "r") as f:
            objectives = json.load(f)

        population.append({
            "id": case_dir.name,
            "origin": "MILP",
            "objectives": objectives,
        })

    return population


# ------------------------------------------------------------
# Plot Pareto fronts (MILP only)
# ------------------------------------------------------------

def plot_pareto_fronts(
    pareto_by_front: dict[int, list[dict]],
    objective_names: tuple[str, str, str],
    output_path: Path
):
    fronts = sorted(pareto_by_front.keys())
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(fronts)))

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])

    pairs = [(1, 0), (1, 2), (0, 2)]

    # ---------- 2D projections ----------
    for idx_pair, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, idx_pair])

        for idx_f, f in enumerate(fronts):
            front = pareto_by_front[f]
            if not front:
                continue

            xs = [p["objectives"][objective_names[i]] for p in front]
            ys = [p["objectives"][objective_names[j]] for p in front]

            ax.scatter(
                xs,
                ys,
                color=colors[idx_f],
                alpha=0.85,
                label=f"F{f}" if idx_pair == 0 else None
            )

        ax.set_xlabel(objective_names[i])
        ax.set_ylabel(objective_names[j])
        ax.set_title(f"{objective_names[i]} vs {objective_names[j]}")
        ax.grid(True)

    # ---------- 3D Pareto ----------
    ax3d = fig.add_subplot(gs[1, 0], projection="3d")

    for idx_f, f in enumerate(fronts):
        front = pareto_by_front[f]
        if not front:
            continue

        xs = [p["objectives"][objective_names[0]] for p in front]
        ys = [p["objectives"][objective_names[1]] for p in front]
        zs = [p["objectives"][objective_names[2]] for p in front]

        ax3d.scatter(
            xs,
            ys,
            zs,
            color=colors[idx_f],
            alpha=0.85,
            label=f"F{f}"
        )

    ax3d.set_xlabel(objective_names[0])
    ax3d.set_ylabel(objective_names[1])
    ax3d.set_zlabel(objective_names[2])
    ax3d.set_title("MILP Pareto Fronts")

    ax3d.legend(
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        title="Fronts"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    population = load_milp_population(MILP_DIR)

    print(f"[INFO] MILP population size: {len(population)}")

    if not population:
        print("[WARN] No MILP solutions found")
        return

    fronts = fast_nondominated_sort(population)
    pareto_by_front = {i: f for i, f in enumerate(fronts)}

    plot_pareto_fronts(
        pareto_by_front,
        OBJECTIVES,
        OUT_PARETO
    )

    print("[OK] MILP Pareto analysis completed")


if __name__ == "__main__":
    main()
