import os, sys, json, time, shutil
from pathlib import Path

import pandas as pd
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient

project_path = os.path.abspath(os.path.join(os.getcwd(), "."))
if project_path not in sys.path:
    sys.path.insert(0, project_path)
    
from lib.cooja_files import convert_simulation_files, convert_cooja_log_to_csv

# ============================================================
# Hardcoded parameters
# ============================================================

INPUT_DIR = Path("./input")
OUTPUT_DIR = Path("./output")

REMOTE_HOST = "localhost"
REMOTE_PORT = 2230
REMOTE_USER = "root"
REMOTE_PASS = "root"

REMOTE_COOJA_DIR = "/opt/contiki-ng/tools/cooja"

JAVA_CMD = (
    "/opt/java/openjdk/bin/java --enable-preview "
    "-Xms4g -Xmx4g "
    "-jar build/libs/cooja.jar --no-gui simulation.csc"
)

TEMPLATE_XML = Path("./simulation_template.xml")

LOCAL_TMP = Path("./tmp")
LOCAL_TMP.mkdir(exist_ok=True)

# ============================================================
# SSH helpers
# ============================================================

def create_ssh() -> SSHClient:
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())
    ssh.connect(
        REMOTE_HOST,
        port=REMOTE_PORT,
        username=REMOTE_USER,
        password=REMOTE_PASS,
    )
    return ssh


def scp_send(ssh: SSHClient, local: Path, remote: str):
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(str(local), remote)


def scp_get(ssh: SSHClient, remote: str, local: Path):
    with SCPClient(ssh.get_transport()) as scp:
        scp.get(remote, str(local))


# ============================================================
# Cooja build
# ============================================================

def build_cooja_simulation_from_json(
    json_path: Path,
    out_dir: Path
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csc = out_dir / "simulation.csc"
    out_dat = out_dir / "positions.dat"

    with open(json_path, "r") as f:
        sim_config = json.load(f)

    convert_simulation_files(
        sim_config,
        TEMPLATE_XML,
        out_csc,
        out_dat
    )

    return {
        "csc": out_csc,
        "positions": out_dat if out_dat.exists() else None,
    }


# ============================================================
# Objective computations
# ============================================================

def mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else float("nan")


def sum_all(df: pd.DataFrame, value_col: str) -> float:
    if value_col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[value_col], errors="coerce")
    return float(s.sum(skipna=True))


def sum_last_minus_first(
    df: pd.DataFrame,
    value_col: str,
    node_col: str = "node",
    time_col: str = "root_time_now",
) -> float:
    if value_col not in df.columns:
        return float("nan")

    df_sorted = df.sort_values([node_col, time_col])
    g = df_sorted.groupby(node_col)[value_col]

    start = pd.to_numeric(g.first(), errors="coerce")
    end = pd.to_numeric(g.last(), errors="coerce")

    per_node = (end - start).clip(lower=0)
    return float(per_node.sum(skipna=True))


def compute_objectives(csv_path: Path) -> dict[str, float]:
    df = pd.read_csv(csv_path)

    return {
        "latency": mean(df["rtt_latency"]) if "rtt_latency" in df.columns else float("nan"),
        "energy": sum_all(df, "total_energy_mj"),
        "throughput": sum_last_minus_first(
            df,
            value_col="server_received",
            node_col="node",
            time_col="root_time_now",
        ),
    }


# ============================================================
# Main pipeline
# ============================================================

def run_simulation(json_file: Path):
    name = json_file.stem
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    build_dir = LOCAL_TMP / name
    if build_dir.exists():
        shutil.rmtree(build_dir)

    files = build_cooja_simulation_from_json(json_file, build_dir)

    ssh = create_ssh()
    try:
        scp_send(ssh, files["csc"], f"{REMOTE_COOJA_DIR}/simulation.csc")
        if files.get("positions"):
            scp_send(ssh, files["positions"], f"{REMOTE_COOJA_DIR}/positions.dat")

        cmd = f"cd {REMOTE_COOJA_DIR} && {JAVA_CMD}"
        stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)

        while not stdout.channel.exit_status_ready():
            time.sleep(0.2)

        log_path = out_dir / "sim.log"
        scp_get(ssh, f"{REMOTE_COOJA_DIR}/COOJA.testlog", log_path)

    finally:
        ssh.close()

    csv_path = out_dir / "sim.csv"
    convert_cooja_log_to_csv(log_path, csv_path)

    objectives = compute_objectives(csv_path)

    with open(out_dir / "objectives.json", "w") as f:
        json.dump(objectives, f, indent=2)

    print(f"[OK] {name} -> {objectives}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for json_file in sorted(INPUT_DIR.glob("*.json")):
        print(f"running {json_file}")
        run_simulation(json_file)


if __name__ == "__main__":
    print("Cooja batch runner")
    main()
