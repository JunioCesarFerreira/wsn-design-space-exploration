import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

MILP_DIR = Path("./simulation/milp-mobile/output")
SIMLAB_FILE = Path("./simlab/generations_p2_csma.json")

OBJECTIVES = ("latency", "energy", "throughput")
MINIMIZE = [True, True, False]  # latency ↓, energy ↓, throughput ↑

OUT_PARETO = Path("pareto_fronts_global.png")

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
# Load MILP results
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
            "id": f"MILP-{case_dir.name}",
            "generation": -1,
            "origin": "MILP",
            "objectives": objectives,
        })

    return population


# ------------------------------------------------------------
# Load SimLab results
# ------------------------------------------------------------

def load_simlab_population(json_path: Path) -> list[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)

    population = []

    for generation, individuals in data.items():
        for ind in individuals:
            population.append({
                "id": ind["simulation_id"],
                "generation": int(generation),
                "origin": "SimLab",
                "objectives": ind["objectives"],
            })

    return population


# ------------------------------------------------------------
# Plot Pareto fronts (MILP highlighted)
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

            for p in front:
                x = p["objectives"][objective_names[i]]
                y = p["objectives"][objective_names[j]]

                if p["origin"] == "MILP":
                    ax.scatter(
                        x, y,
                        marker="*",
                        s=200,
                        facecolors=colors[idx_f],  
                        edgecolors="black",
                        linewidths=1.0,
                        zorder=10,
                    )
                else:
                    ax.scatter(
                        x, y,
                        color=colors[idx_f],
                        alpha=0.8
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

        for p in front:
            x, y, z = (
                p["objectives"][objective_names[0]],
                p["objectives"][objective_names[1]],
                p["objectives"][objective_names[2]],
            )

            if p["origin"] == "MILP":
                ax3d.scatter(
                    x, y, z,
                    marker="*",
                    s=260,
                    facecolors=colors[idx_f],  
                    edgecolors="black",
                    linewidths=1.0,
                    zorder=10,
                )
            else:
                ax3d.scatter(
                    x, y, z,
                    color=colors[idx_f],
                    alpha=0.8
                )

    ax3d.set_xlabel(objective_names[0])
    ax3d.set_ylabel(objective_names[1])
    ax3d.set_zlabel(objective_names[2])
    ax3d.set_title("Global Pareto Fronts (MILP highlighted)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    milp_pop = load_milp_population(MILP_DIR)
    simlab_pop = load_simlab_population(SIMLAB_FILE)

    population = milp_pop + simlab_pop

    print(f"[INFO] Population size: {len(population)}")
    print(f"[INFO] MILP: {len(milp_pop)} | SimLab: {len(simlab_pop)}")

    fronts = fast_nondominated_sort(population)
    pareto_by_front = {i: f for i, f in enumerate(fronts)}

    plot_pareto_fronts(
        pareto_by_front,
        OBJECTIVES,
        OUT_PARETO
    )

    print("[OK] Global Pareto analysis completed")


if __name__ == "__main__":
    main()
