import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ======================
# Helper: trajetórias com quebras
# ======================

def _traj_with_breaks(r_mobile, name, T, close=False, jump_factor=5.0):
    """
    Gera uma trajetória shape (N,2) com linhas 'quebradas' (NaNs) em saltos grandes.
    - close=False por padrão: não faz o wrap (evita risco entre último e primeiro).
    - jump_factor controla a sensibilidade: quebra quando passo > jump_factor * mediana.
    """
    pts = np.array([r_mobile(name, t) for t in range(1, T + 1)], dtype=float)

    # Opcionalmente fechar (em geral, NÃO faça para caminhos abertos)
    if close:
        pts = np.vstack([pts, pts[0]])

    # Distâncias entre pontos consecutivos
    if len(pts) >= 2:
        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        med = np.median(dists) if np.any(dists > 0) else 0.0
        thr = max(1e-9, med * jump_factor)  # limiar robusto

        # Monta com NaNs onde houver salto
        rows = [pts[0]]
        for k in range(1, len(pts)):
            if dists[k - 1] > thr:
                rows.append([np.nan, np.nan])  # quebra a linha aqui
            rows.append(pts[k])
        pts = np.array(rows, dtype=float)
    return pts

# ======================
# Paleta / tamanhos (padrão visual)
# ======================

COLOR_SINK   = "blue"
COLOR_CANDIDATE = "black"
COLOR_FIXED0 = "gray"    # não instalado
COLOR_FIXED1 = "red"     # instalado
COLOR_MOBILE = "green"

S_SINK   = 260
S_FIXED0 = 40
S_FIXED1 = 120
S_MOBILE = 18  # pontos da trajetória

# ================
# FIGURA 1
# ================

def plot_candidates_and_paths(F, q_fixed, q_sink, R_comm, mob_names, r_mobile, 
                              T, region, out_path="./pic1.jpg"):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # candidatos + raios
    for j in F:
        q = q_fixed[j]
        ax.add_patch(Circle((q[0], q[1]), R_comm, fill=False, linewidth=1, ls='--', edgecolor="gray"))
        ax.scatter([q[0]], [q[1]], marker='s', s=S_FIXED0, c=COLOR_CANDIDATE)

    # sink + raio
    ax.scatter([q_sink[0]], [q_sink[1]], marker='*', s=S_SINK, c=COLOR_SINK, label="sink")
    ax.add_patch(Circle((q_sink[0], q_sink[1]), R_comm, fill=False, linewidth=1, ls='--', edgecolor=COLOR_SINK, alpha=0.6))

    # trajetórias (NÃO fecha; quebra saltos com NaN)
    for name in mob_names:
        traj = _traj_with_breaks(r_mobile, name, T, close=False, jump_factor=5.0)
        ax.plot(traj[:, 0], traj[:, 1], linestyle='--', linewidth=2, alpha=0.6, c=COLOR_MOBILE, label=None)
        ax.scatter(traj[:, 0], traj[:, 1], marker='o', s=S_MOBILE, c=COLOR_MOBILE, alpha=0.7, label=None)

    ax.set_title("Candidatos, sink e trajetórias (raios de comunicação)")
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc="best")
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ================
# FIGURA 2
# ================

def plot_solution(F, installed, q_fixed, q_sink, R_comm, R_inter,
                  mob_names, T, r_mobile, region, out_path="./pic2.png"):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # todos candidatos
    installed_set = set(installed)

    for j in F:
        if j not in installed_set:
            q = q_fixed[j]
            ax.scatter([q[0]], [q[1]], marker='s', s=S_FIXED0, c=COLOR_FIXED0, alpha=0.9)

    # instalados
    for j in installed:
        q = q_fixed[j]
        ax.scatter([q[0]], [q[1]], marker='s', s=S_FIXED1, c=COLOR_FIXED1, label=f"instalado {j[1]}")
        ax.add_patch(Circle((q[0], q[1]), R_comm, fill=False, linewidth=1, ls='--', edgecolor=COLOR_FIXED1, alpha=0.6))
        ax.add_patch(Circle((q[0], q[1]), R_inter, fill=False, linewidth=1, ls='--', edgecolor="orange", alpha=0.6))

    # sink
    ax.scatter([q_sink[0]], [q_sink[1]], marker='*', s=S_SINK, c=COLOR_SINK, label="sink")
    ax.add_patch(Circle((q_sink[0], q_sink[1]), R_comm, fill=False, linewidth=1, ls='--', edgecolor=COLOR_SINK, alpha=0.6))
    ax.add_patch(Circle((q_sink[0], q_sink[1]), R_inter, fill=False, linewidth=1, ls='--', edgecolor="orange", alpha=0.6))

    # trajetórias (contexto): não fechar, com quebras
    for name in mob_names:
        traj = _traj_with_breaks(r_mobile, name, T, close=False, jump_factor=5.0)
        ax.plot(traj[:, 0], traj[:, 1], linestyle='--', linewidth=2, alpha=0.6, c=COLOR_MOBILE)
        ax.scatter(traj[:, 0], traj[:, 1], marker='o', s=S_MOBILE, c=COLOR_MOBILE, alpha=0.7)

    ax.set_title(f"Solução (raios de comunicação instalados)")
    ax.axis('equal')
    ax.grid(True)
    #ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_installed_graph(installed, q_fixed, q_sink, R_comm, region, out_path="./pic_installed_graph.png"):
    """
    Plota apenas o grafo formado pelos motes FIXOS instalados.
    - Vértices: instalados (COLOR_FIXED1) e sink (COLOR_SINK).
    - Arestas: entre instalados (e do sink para instalados) quando ||u - v|| <= R_comm.
    - Sem móveis e sem candidatos não instalados.
    """
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # nós instalados (vermelho, conforme seu padrão atual)
    for j in installed:
        q = q_fixed[j]
        ax.scatter([q[0]], [q[1]], marker='s', s=S_FIXED1, c=COLOR_FIXED1)

    # sink (estrela azul)
    ax.scatter([q_sink[0]], [q_sink[1]], marker='*', s=S_SINK, c=COLOR_SINK)

    # arestas (vermelhas contínuas) entre instalados
    def _dist(p, q):
        return float(np.hypot(p[0]-q[0], p[1]-q[1]))

    # entre instalados
    for idx in range(len(installed)):
        for jdx in range(idx + 1, len(installed)):
            a = installed[idx]; b = installed[jdx]
            pa, pb = q_fixed[a], q_fixed[b]
            if _dist(pa, pb) <= R_comm + 1e-9:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], linewidth=2.0, linestyle='-', c="red")

    # sink ↔ instalados
    for j in installed:
        p = q_fixed[j]
        if _dist(q_sink, p) <= R_comm + 1e-9:
            ax.plot([q_sink[0], p[0]], [q_sink[1], p[1]], linewidth=2.0, linestyle='-', c="red")

    ax.set_title("Grafo dos fixos instalados (arestas ≤ R_comm)")
    ax.axis('equal')
    ax.grid(True)
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
