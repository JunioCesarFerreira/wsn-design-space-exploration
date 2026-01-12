import os, io, shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "Este script requer 'Pillow' para gerar o GIF. Instale com: pip install pillow"
    ) from e 
    

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
        # limiar robusto: maior entre 1e-9 e mediana*jump_factor
        thr = max(1e-9, med * jump_factor)

        # Monta com NaNs onde houver salto
        rows = [pts[0]]
        for k in range(1, len(pts)):
            if dists[k - 1] > thr:
                rows.append([np.nan, np.nan])  # quebra a linha aqui
            rows.append(pts[k])
        pts = np.array(rows, dtype=float)
    return pts    


def save_routes_gif(
    installed, r_mobile, mob_names, q_sink, q_fixed, R_comm, region,
    x_val, E_t, T, F, out_dir_path: Path, *, jump_factor: float = 5.0, fps: int = 3
):
    # Paleta / estilo
    COLOR_SINK   = "blue"
    COLOR_FIXED0 = "gray"    # não instalado
    COLOR_FIXED1 = "green"   # instalado
    COLOR_MOBILE = "black"
    COLOR_LINK   = "red"

    S_SINK   = 260
    S_FIXED0 = 40
    S_FIXED1 = 120
    S_MOBILE = 80

    # Pasta de frames
    frames_dir = out_dir_path / "frames_gif"
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    FLOW_EPS = 1e-6

    # Trajetórias com quebras
    traj_cache = {
        name: _traj_with_breaks(r_mobile, name, T, close=False, jump_factor=jump_factor)
        for name in mob_names
    }

    def _pos_node(n, t):
        if n[0] == "sink":
            return q_sink
        if n[0] == "j":
            return q_fixed[n]
        if n[0] == "m":
            return r_mobile(n[1], t)
        raise ValueError("nó desconhecido")

    def _draw_frame(t):
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # Fixos não instalados (cinza)
        installed_set = set(installed)
        for j in F:
            if j not in installed_set:
                q = q_fixed[j]
                ax.scatter([q[0]], [q[1]], marker='s', s=S_FIXED0, c=COLOR_FIXED0, alpha=0.9)

        # Fixos instalados (verde) + (se quiser, mantenha o círculo de alcance)
        for j in installed:
            q = q_fixed[j]
            ax.scatter([q[0]], [q[1]], marker='s', s=S_FIXED1, c=COLOR_FIXED1)
            ax.add_patch(Circle((q[0], q[1]), R_comm, fill=False, linewidth=1, ls='--', edgecolor=COLOR_FIXED1, alpha=0.6))

        # Sink (estrela azul)
        ax.scatter([q_sink[0]], [q_sink[1]], marker='*', s=S_SINK, c=COLOR_SINK)

        # Trajetórias (pretas, tracejadas e leves para contexto)
        for name in mob_names:
            traj = traj_cache[name]
            ax.plot(traj[:, 0], traj[:, 1], linestyle=':', linewidth=2, alpha=0.5, c=COLOR_MOBILE)

        # Móveis (pretos)
        for name in mob_names:
            pm = r_mobile(name, t)
            ax.scatter([pm[0]], [pm[1]], marker='o', s=S_MOBILE, c=COLOR_MOBILE)

        # Links ativos (vermelho sólido)
        flows = {
            (i, j): x_val.get((i, j, t), 0.0)
            for (i, j) in E_t.get(t, [])
            if x_val.get((i, j, t), 0.0) > FLOW_EPS
        }
        for (i, j), val in flows.items():
            pi, pj = _pos_node(i, t), _pos_node(j, t)
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], linewidth=2.2, linestyle='-', c=COLOR_LINK, alpha=0.95)

        ax.set_title(f"Rotas de comunicação (t = {t})")
        ax.axis('equal')
        ax.grid(True)
        if region and len(region) == 4:
            ax.set_xlim(region[0], region[2])
            ax.set_ylim(region[1], region[3])
        plt.tight_layout()

        buf = io.BytesIO()
        fig.canvas.print_png(buf)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("P")
        return img

    frames = []
    for t in range(1, T + 1):
        img = _draw_frame(t)
        frame_path = os.path.join(frames_dir, f"frame_{t:03d}.png")
        img.convert("RGB").save(frame_path, format="PNG")
        frames.append(img)

    gif_path = out_dir_path / "routes.gif"
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / max(1, fps)),
            loop=0
        )
    return gif_path


        
def save_routes2_gif(
    installed, r_mobile, mob_names, q_sink, q_fixed, R_comm, region,
    x_val, E_t, T, F, out_dir_path: Path, *, jump_factor: float = 5.0, fps: int = 3
):
    # Paleta / estilo
    COLOR_SINK   = "blue"
    COLOR_FIXED0 = "gray"    # não instalado
    COLOR_FIXED1 = "green"   # instalado
    COLOR_MOBILE = "black"
    COLOR_LINK   = "red"

    S_SINK   = 260
    S_FIXED0 = 40
    S_FIXED1 = 120
    S_MOBILE = 80

    # Pasta de frames
    frames_dir = out_dir_path / "frames_gif"
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    FLOW_EPS = 1e-6

    # Trajetórias com quebras
    traj_cache = {
        name: _traj_with_breaks(r_mobile, name, T, close=False, jump_factor=jump_factor)
        for name in mob_names
    }

    installed_set = set(installed)

    def _pos_node(n, t):
        if n[0] == "sink":
            return q_sink
        if n[0] == "j":
            return q_fixed[n]
        if n[0] == "m":
            return r_mobile(n[1], t)
        raise ValueError("nó desconhecido")

    def _draw_frame(t):
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # Fixos não instalados (cinza)
        for j in F:
            if j not in installed_set:
                q = q_fixed[j]
                ax.scatter([q[0]], [q[1]], marker='s', s=S_FIXED0, c=COLOR_FIXED0, alpha=0.9)

        # Fixos instalados (verde)
        for j in installed:
            q = q_fixed[j]
            ax.scatter([q[0]], [q[1]], marker='s', s=S_FIXED1, c=COLOR_FIXED1)

        # Sink (estrela azul)
        ax.scatter([q_sink[0]], [q_sink[1]], marker='*', s=S_SINK, c=COLOR_SINK)

        # Trajetórias dos móveis (pretas tracejadas)
        for name in mob_names:
            traj = traj_cache[name]
            ax.plot(traj[:, 0], traj[:, 1], linestyle=':', linewidth=2, alpha=0.6, c=COLOR_MOBILE)

        # Posição atual de cada móvel (pretos) + círculo de alcance (opcional, preto tracejado)
        for name in mob_names:
            pm = r_mobile(name, t)
            ax.scatter([pm[0]], [pm[1]], marker='o', s=S_MOBILE, c=COLOR_MOBILE)
            ax.add_patch(Circle((pm[0], pm[1]), R_comm, fill=False, linewidth=1, ls='--', edgecolor=COLOR_MOBILE, alpha=0.6))

        # Links ativos (vermelho sólido, sem distinção)
        flows = {
            (i, j): x_val.get((i, j, t), 0.0)
            for (i, j) in E_t.get(t, [])
            if x_val.get((i, j, t), 0.0) > FLOW_EPS
        }
        for (i, j), val in flows.items():
            pi, pj = _pos_node(i, t), _pos_node(j, t)
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], linewidth=2.4, linestyle='-', c=COLOR_LINK, alpha=0.95)

        ax.set_title(f"Rotas de comunicação (t = {t})")
        ax.axis('equal')
        ax.grid(True)
        if region and len(region) == 4:
            ax.set_xlim(region[0], region[2])
            ax.set_ylim(region[1], region[3])
        plt.tight_layout()

        buf = io.BytesIO()
        fig.canvas.print_png(buf)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("P")
        return img

    frames = []
    for t in range(1, T + 1):
        img = _draw_frame(t)
        frame_path = os.path.join(frames_dir, f"frame_{t:03d}.png")
        img.convert("RGB").save(frame_path, format="PNG")
        frames.append(img)

    gif_path = out_dir_path / "routes2.gif"
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / max(1, fps)),
            loop=0
        )
    return gif_path

