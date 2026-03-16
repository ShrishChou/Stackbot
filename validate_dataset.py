import os
import glob
import numpy as np

DATASET_DIR = "datasets"

REQUIRED_KEYS = [
    "obs",
    "actions",
    "next_obs",
    "dones",
    "success",
]

# Based on your current collection format:
# obs = [qpos7, qvel7, finger2, site_pos3, cube_a_pos3, cube_b_pos3, cube_a-site3, cube_b-cube_a3]
EXPECTED_OBS_DIM = 7 + 7 + 2 + 3 + 3 + 3 + 3 + 3  # 31
EXPECTED_ACT_DIM = 8  # 7 arm controls + 1 gripper control

# Panda actuator control ranges from your XML
ARM_CTRL_RANGES = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
], dtype=float)

GRIP_MIN = 0.0
GRIP_MAX = 255.0

# Observation layout slices
QPOS_SLICE = slice(0, 7)
QVEL_SLICE = slice(7, 14)
FINGER_SLICE = slice(14, 16)
SITE_POS_SLICE = slice(16, 19)
CUBE_A_POS_SLICE = slice(19, 22)
CUBE_B_POS_SLICE = slice(22, 25)
CUBE_A_MINUS_SITE_SLICE = slice(25, 28)
CUBE_B_MINUS_CUBE_A_SLICE = slice(28, 31)


def check_no_nan_inf(name, arr, errors):
    if not np.all(np.isfinite(arr)):
        errors.append(f"{name} contains NaN or inf")


def check_shape(name, arr, ndim, errors):
    if arr.ndim != ndim:
        errors.append(f"{name} has ndim={arr.ndim}, expected {ndim}")


def validate_single_file(path):
    errors = []
    warnings = []

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        return [f"failed to load: {e}"], warnings

    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"missing required key '{key}'")

    if errors:
        return errors, warnings

    obs = data["obs"]
    actions = data["actions"]
    next_obs = data["next_obs"]
    dones = data["dones"]
    success = data["success"]

    check_shape("obs", obs, 2, errors)
    check_shape("actions", actions, 2, errors)
    check_shape("next_obs", next_obs, 2, errors)

    if dones.ndim not in (1, 2):
        errors.append(f"dones has ndim={dones.ndim}, expected 1 or 2")

    if success.ndim not in (0, 1):
        errors.append(f"success has ndim={success.ndim}, expected 0 or 1")

    if errors:
        return errors, warnings

    T = obs.shape[0]

    if actions.shape[0] != T:
        errors.append(f"actions length {actions.shape[0]} != obs length {T}")
    if next_obs.shape[0] != T:
        errors.append(f"next_obs length {next_obs.shape[0]} != obs length {T}")

    dones_flat = dones.reshape(-1)
    if dones_flat.shape[0] != T:
        errors.append(f"dones length {dones_flat.shape[0]} != obs length {T}")

    if obs.shape[1] != EXPECTED_OBS_DIM:
        errors.append(f"obs dim {obs.shape[1]} != expected {EXPECTED_OBS_DIM}")
    if next_obs.shape[1] != EXPECTED_OBS_DIM:
        errors.append(f"next_obs dim {next_obs.shape[1]} != expected {EXPECTED_OBS_DIM}")
    if actions.shape[1] != EXPECTED_ACT_DIM:
        errors.append(f"actions dim {actions.shape[1]} != expected {EXPECTED_ACT_DIM}")

    check_no_nan_inf("obs", obs, errors)
    check_no_nan_inf("actions", actions, errors)
    check_no_nan_inf("next_obs", next_obs, errors)

    if errors:
        return errors, warnings

    # Episode length metadata
    if "episode_len" in data:
        episode_len = int(np.array(data["episode_len"]).reshape(-1)[0])
        if episode_len != T:
            errors.append(f"episode_len={episode_len} but obs length={T}")

    # done sanity
    if T == 0:
        errors.append("empty trajectory")
        return errors, warnings

    if np.sum(dones_flat.astype(np.int32)) > 1:
        warnings.append("more than one done=True in trajectory")

    if not bool(dones_flat[-1]):
        warnings.append("last done is not True")

    # success sanity
    success_value = bool(np.array(success).reshape(-1)[0])
    if success_value and not bool(dones_flat[-1]):
        warnings.append("success=True but final done is not True")

    # Joint target range sanity
    arm_actions = actions[:, :7]
    grip_actions = actions[:, 7]

    for j in range(7):
        low, high = ARM_CTRL_RANGES[j]
        below = np.sum(arm_actions[:, j] < low - 1e-6)
        above = np.sum(arm_actions[:, j] > high + 1e-6)
        if below or above:
            errors.append(
                f"joint action {j} out of range {low:.4f}..{high:.4f} "
                f"(below={below}, above={above})"
            )

    if np.any(grip_actions < GRIP_MIN - 1e-6) or np.any(grip_actions > GRIP_MAX + 1e-6):
        errors.append("gripper action out of [0, 255] range")

    # Observation consistency checks
    cube_a_minus_site = obs[:, CUBE_A_MINUS_SITE_SLICE]
    recomputed_cube_a_minus_site = obs[:, CUBE_A_POS_SLICE] - obs[:, SITE_POS_SLICE]
    if not np.allclose(cube_a_minus_site, recomputed_cube_a_minus_site, atol=1e-4):
        warnings.append("cube_a_pos - site_pos slice does not match stored relative feature")

    cube_b_minus_cube_a = obs[:, CUBE_B_MINUS_CUBE_A_SLICE]
    recomputed_cube_b_minus_cube_a = obs[:, CUBE_B_POS_SLICE] - obs[:, CUBE_A_POS_SLICE]
    if not np.allclose(cube_b_minus_cube_a, recomputed_cube_b_minus_cube_a, atol=1e-4):
        warnings.append("cube_b_pos - cube_a_pos slice does not match stored relative feature")

    # Finger sanity
    fingers = obs[:, FINGER_SLICE]
    if np.any(fingers < -0.01) or np.any(fingers > 0.05):
        warnings.append("finger joint values outside expected rough range [-0.01, 0.05]")

    # Optional final stack sanity if success=True
    if success_value:
        final_obs = next_obs[-1]
        cube_a_final = final_obs[CUBE_A_POS_SLICE]
        cube_b_final = final_obs[CUBE_B_POS_SLICE]
        xy_dist = np.linalg.norm(cube_a_final[:2] - cube_b_final[:2])
        dz = cube_a_final[2] - cube_b_final[2]

        if xy_dist > 0.05:
            warnings.append(f"success=True but final XY cube distance is large ({xy_dist:.4f})")
        if not (0.02 <= dz <= 0.09):
            warnings.append(f"success=True but final cube height difference looks odd ({dz:.4f})")

    return errors, warnings


def main():
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.npz")))
    if not files:
        print(f"No .npz files found in {DATASET_DIR}")
        return

    total_errors = 0
    total_warnings = 0
    valid_files = 0

    print(f"Checking {len(files)} trajectory files in '{DATASET_DIR}'...\n")

    for path in files:
        errors, warnings = validate_single_file(path)
        rel = os.path.basename(path)

        if not errors and not warnings:
            print(f"[OK]   {rel}")
            valid_files += 1
            continue

        if errors:
            print(f"[FAIL] {rel}")
            for e in errors:
                print(f"       error: {e}")
            total_errors += len(errors)
        else:
            print(f"[WARN] {rel}")

        for w in warnings:
            print(f"       warn:  {w}")
        total_warnings += len(warnings)

        if not errors:
            valid_files += 1

    print("\nSummary")
    print(f"  files checked:   {len(files)}")
    print(f"  structurally ok: {valid_files}")
    print(f"  total warnings:  {total_warnings}")
    print(f"  total errors:    {total_errors}")


if __name__ == "__main__":
    main()