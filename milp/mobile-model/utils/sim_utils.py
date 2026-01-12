# sim_utils.py
import json
import math
import numpy as np

def load_simulation_json(path_or_dict):
    """Aceita um caminho de arquivo .json OU um dict já carregado e retorna o bloco 'simulationModel'."""
    if isinstance(path_or_dict, str):
        with open(path_or_dict, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(path_or_dict, dict):
        data = path_or_dict
    else:
        raise TypeError("path_or_dict deve ser str (caminho) ou dict (JSON carregado).")
    return data["simulationModel"]

def _safe_eval_expr(expr: str, t: float) -> float:
    """
    Avalia expressão de string do JSON com variável escalar 't' em [0,1].
    Permite np.* e números; sem builtins perigosos.
    """
    try:
        return float(expr)
    except (ValueError, TypeError):
        pass
    allowed_globals = {"__builtins__": {}, "np": np, "math": math}
    allowed_locals = {"t": float(t)}
    return float(eval(expr, allowed_globals, allowed_locals))

def _segment_length(function_pair, nsamples=200):
    """
    Aproxima o comprimento do segmento definido por (x_expr, y_expr), t in [0,1],
    por uma polilinha com 'nsamples' amostras.
    """
    x_expr, y_expr = function_pair
    ts = np.linspace(0.0, 1.0, nsamples + 1)
    pts = np.stack([
        [_safe_eval_expr(str(x_expr), tt), _safe_eval_expr(str(y_expr), tt)]
        for tt in ts
    ], axis=0)
    diffs = np.diff(pts, axis=0)
    seg_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    return seg_len

def _distribute_integer_proportions(total_steps, weights):
    """
    Dado total_steps (int) e uma lista/array de pesos não-negativos,
    retorna uma lista de inteiros que somam total_steps, proporcional aos pesos.
    Estratégia: floor + atribuição dos restos de maiores frações.
    """
    w = np.asarray(weights, dtype=float)
    if len(w) == 0:
        return []
    if np.all(w <= 0):
        base = total_steps // len(w)
        rem  = total_steps % len(w)
        steps = [base] * len(w)
        for i in range(rem):
            steps[i] += 1
        return steps
    w = np.maximum(w, 0.0)
    W = float(np.sum(w))
    raw = (total_steps * w / W) if W > 0 else np.zeros_like(w)
    flo = np.floor(raw).astype(int)
    rem = int(total_steps - np.sum(flo))
    frac = raw - flo
    order = np.argsort(-frac)
    steps = flo.tolist()
    for k in range(rem):
        steps[order[k]] += 1
    return steps

def make_mobile_trajectory_fn(function_path, is_closed: bool, is_roundtrip: bool, T: int, speed: float):
    """
    Cria r(tau) -> R^2 para tau=1..T.
    - Se is_closed=True: percorre os segmentos em ciclo (apenas "ida", orientação direta).
    - Se is_roundtrip=True e não for fechado: faz vai-e-volta ponto a ponto:
        [0,1,2,...,K-1, K-1,...,2,1,0] onde os da 'volta' são percorridos no sentido inverso.
    A distribuição de passos por segmento é proporcional ao tempo (len / speed).
    """
    K = len(function_path)
    if K == 0:
        raise ValueError("functionPath vazio.")

    # Comprimento por segmento original (param t ∈ [0,1], orientação direta)
    lens_by_k = [_segment_length(function_path[k], nsamples=200) for k in range(K)]

    # Sequência efetiva de (segmento, direcao) onde direcao = +1 (ida) ou -1 (volta)
    seq = []
    if is_closed:
        # ciclo apenas no sentido direto
        seq = [(k, +1) for k in range(K)]
    elif is_roundtrip and K >= 1:
        # ida
        seq.extend((k, +1) for k in range(K))
        # volta (espelha todos os segmentos no sentido inverso)
        # Obs.: para evitar duplicar "cantos" demais, a ordem abaixo inclui todos;
        # se quiser eliminar um endpoint duplicado, pode trocar o range para (K-2...0).
        seq.extend((k, -1) for k in range(K - 1, -1, -1))
    else:
        # caminho aberto somente no sentido direto
        seq = [(k, +1) for k in range(K)]

    # Tempo efetivo por (segmento,direção): igual ao do segmento
    spd = 1.0 if speed is None or speed <= 0 else float(speed)
    times_eff = [lens_by_k[k] / spd for (k, _dir) in seq]

    # Alocação discreta de passos
    steps_per_leg = _distribute_integer_proportions(T, times_eff)

    # Garante pelo menos 1 passo por perna quando fizer sentido
    if T >= len(seq):
        steps_per_leg = [max(1, s) for s in steps_per_leg]
        surplus = int(sum(steps_per_leg) - T)
        if surplus > 0:
            order = np.argsort(times_eff)  # remove dos mais curtos primeiro
            for idx in order:
                if surplus == 0: break
                if steps_per_leg[idx] > 1:
                    steps_per_leg[idx] -= 1
                    surplus -= 1
        elif surplus < 0:
            deficit = -surplus
            order = np.argsort(-np.asarray(times_eff))  # adiciona nos mais longos
            for k in range(deficit):
                steps_per_leg[order[k % len(seq)]] += 1

    cut = np.cumsum([0] + steps_per_leg)
    S = len(seq)

    def r_of_tau(tau: int) -> np.ndarray:
        u = tau - 1
        leg = int(np.searchsorted(cut, u, side="right") - 1)
        leg = min(max(leg, 0), S - 1)
        local_len = steps_per_leg[leg]
        if local_len <= 1:
            tloc = 1.0  # ponto final do param
        else:
            tloc = (u - cut[leg]) / (local_len - 1)  # ∈ [0,1]
        k, direc = seq[leg]

        # t efetivo conforme a direção: ida = t, volta = 1 - t
        teff = tloc if direc == +1 else (1.0 - tloc)

        x_expr, y_expr = function_path[k]
        x = _safe_eval_expr(str(x_expr), teff)
        y = _safe_eval_expr(str(y_expr), teff)
        return np.array([x, y], dtype=float)

    return r_of_tau


# ==============================
# Fixos aleatórios
# ==============================

def add_random_fixed_motes(sim_model: dict, n_new: int = 20, seed: int = 42) -> dict:
    """
    Adiciona n_new motes fixos aleatórios dentro de simulationModel['region'].
    Mantém 'root' e continua a numeração 'nodeX' a partir do maior X existente.
    Opera diretamente sobre o dict 'simulationModel' (retorna o mesmo objeto mutado).
    """
    xmin, ymin, xmax, ymax = map(float, sim_model["region"])
    fixed = sim_model["simulationElements"]["fixedMotes"]

    # nomes existentes e maior índice nodeNN
    import re
    names = {fm["name"] for fm in fixed}
    pat = re.compile(r"^node(\d+)$", re.IGNORECASE)
    max_idx = 0
    for nm in names:
        m = pat.match(nm)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    rng = np.random.default_rng(seed)
    for _ in range(n_new):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        max_idx += 1
        new_name = f"node{max_idx}"
        while new_name in names:
            max_idx += 1
            new_name = f"node{max_idx}"
        new_fixed = {
            "position": [float(x), float(y)],
            "name": new_name,
            "sourceCode": "node.c"
        }
        fixed.append(new_fixed)
        names.add(new_name)
    return sim_model