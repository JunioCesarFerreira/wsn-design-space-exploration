import logging
import numpy as np

def evaluate_function(expression: str, t_values: np.ndarray) -> np.ndarray:
    return np.array([eval(expression, {"t": t, "np": np}) for t in t_values])

def generate_positions_from_json(
    simElements: dict, 
    output_filename: str = "positions.dat",
    debug = False
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:

    fixed_positions = [(mote["position"][0], mote["position"][1]) for mote in simElements["fixedMotes"]]
    mobile_motes = simElements["mobileMotes"]
    
    mobile_start_positions = []

    with open(output_filename, "w") as file:
        file.write("# Fixed positions\n")
        for i, (x, y) in enumerate(fixed_positions):
            file.write(f"{i} 0.00000000 {x:.2f} {y:.2f}\n")
        file.write("\n")

        file.write("# Mobile nodes\n")
        mote_index = len(fixed_positions)
        max_steps = 0
        mobile_trajectories = []

        for mote in mobile_motes:
            path_segments = mote["functionPath"]
            speed = mote["speed"]
            time_step = mote["timeStep"]
            is_round_trip = mote.get("isRoundTrip", False)
            
            if debug:
                log = logging.getLogger(__name__)
                log.debug("path_segments", path_segments)

            # Avaliação dos segmentos
            x_all, y_all, segment_distances = [], [], []
            for x_expr, y_expr in path_segments:
                t_values = np.linspace(0, 1, num=100)
                x_vals = evaluate_function(x_expr, t_values)
                y_vals = evaluate_function(y_expr, t_values)
                x_all.append(x_vals)
                y_all.append(y_vals)
                segment_distances.append(np.sum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)))

            total_distance = np.sum(segment_distances)
            total_duration = total_distance / speed
            total_steps = max(1, int(total_duration / time_step))
            max_steps = max(max_steps, total_steps)

            # Interpolação proporcional por segmento
            x_full, y_full = [], []
            for x_vals, y_vals, seg_dist in zip(x_all, y_all, segment_distances):
                proportion = seg_dist / total_distance if total_distance > 0 else 1
                seg_steps = max(1, int(proportion * total_steps))
                interp_t = np.linspace(0, 1, seg_steps)
                x_interp = np.interp(interp_t, np.linspace(0, 1, len(x_vals)), x_vals)
                y_interp = np.interp(interp_t, np.linspace(0, 1, len(y_vals)), y_vals)
                x_full.extend(x_interp)
                y_full.extend(y_interp)

            x_full = np.array(x_full)
            y_full = np.array(y_full)

            if is_round_trip:
                x_full = np.concatenate((x_full, x_full[::-1]))
                y_full = np.concatenate((y_full, y_full[::-1]))

            mobile_start_positions.append((x_full[0], y_full[0]))
            mobile_trajectories.append((mote_index, x_full, y_full, time_step))
            mote_index += 1

        for step in range(2 * max_steps):
            for mote_id, x_full, y_full, time_step in mobile_trajectories:
                if step < len(x_full):
                    file.write(f"{mote_id} {step * time_step:.8f} {x_full[step]:.2f} {y_full[step]:.2f}\n")
            file.write("\n")

    return fixed_positions, mobile_start_positions
