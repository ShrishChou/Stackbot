import os
import time
import numpy as np
import mujoco

MODEL_PATH = "franka_emika_panda/stack_scene.xml"
SAVE_DIR = "datasets"

HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255], dtype=float)

ARM_DOF = 7
GRIP_OPEN = 255.0
GRIP_CLOSED = 0.0

SITE_NAME = "grasp_site"
CUBE_A_BODY = "cube_a"
CUBE_B_BODY = "cube_b"
CUBE_A_JOINT = "cube_a_freejoint"
CUBE_B_JOINT = "cube_b_freejoint"


def get_id(model, objtype, name):
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx == -1:
        raise ValueError(f"Could not find {name}")
    return idx


def near(a, b, tol=0.01):
    return np.linalg.norm(a - b) < tol


def site_rotmat(data, site_id):
    return data.site_xmat[site_id].reshape(3, 3).copy()


def mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(q1, q2):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q1, q2)
    return out


def orientation_error(current_R, target_R):
    current_q = mat_to_quat(current_R)
    target_q = mat_to_quat(target_R)
    q_err = quat_mul(target_q, quat_conj(current_q))
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def pose_ik_step(
    model,
    data,
    site_id,
    target_pos,
    target_R,
    home_q,
    step_size=0.35,
    damping=1e-2,
    rot_weight=2.0,
    posture_gain=0.03,
):
    current_pos = data.site_xpos[site_id].copy()
    current_R = site_rotmat(data, site_id)

    pos_err = target_pos - current_pos
    rot_err = orientation_error(current_R, target_R)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    Jp = jacp[:, :ARM_DOF]
    Jr = jacr[:, :ARM_DOF]

    J = np.vstack([Jp, rot_weight * Jr])
    err = np.concatenate([pos_err, rot_weight * rot_err])

    JT = J.T
    dq_task = JT @ np.linalg.solve(J @ JT + damping * np.eye(6), err)

    q = data.qpos[:ARM_DOF].copy()
    J_pinv = np.linalg.pinv(J)
    N = np.eye(ARM_DOF) - J_pinv @ J
    dq_posture = posture_gain * (home_q - q)

    dq = dq_task + N @ dq_posture
    q_des = q + step_size * dq

    for i in range(ARM_DOF):
        low, high = model.actuator_ctrlrange[i]
        q_des[i] = np.clip(q_des[i], low, high)

    return q_des, pos_err, rot_err


def get_obs(model, data, site_id, cube_a_id, cube_b_id):
    site_pos = data.site_xpos[site_id].copy()
    cube_a_pos = data.xpos[cube_a_id].copy()
    cube_b_pos = data.xpos[cube_b_id].copy()

    obs = np.concatenate([
        data.qpos[:7].copy(),
        data.qvel[:7].copy(),
        data.qpos[7:9].copy(),
        site_pos,
        cube_a_pos,
        cube_b_pos,
        cube_a_pos - site_pos,
        cube_b_pos - cube_a_pos,
    ])
    return obs


def check_success(data, cube_a_id, cube_b_id, xy_tol=0.03, z_tol=0.03, expected_dz=0.05):
    cube_a = data.xpos[cube_a_id].copy()
    cube_b = data.xpos[cube_b_id].copy()

    xy_ok = np.linalg.norm(cube_a[:2] - cube_b[:2]) < xy_tol
    z_ok = abs((cube_a[2] - cube_b[2]) - expected_dz) < z_tol
    return xy_ok and z_ok


def reset_robot(model, data):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.act[:] = 0.0 if model.na > 0 else data.act[:]

    # Panda home qpos: 7 arm + 2 finger joints
    data.qpos[:9] = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04], dtype=float)
    data.ctrl[:] = HOME_CTRL
    mujoco.mj_forward(model, data)

    for _ in range(200):
        data.ctrl[:] = HOME_CTRL
        mujoco.mj_step(model, data)


def reset_cubes(model, data, cube_a_qpos_adr, cube_b_qpos_adr, cube_a_pos, cube_b_pos):
    # freejoint qpos format: x y z qw qx qy qz
    data.qpos[cube_a_qpos_adr:cube_a_qpos_adr+7] = np.array([
        cube_a_pos[0], cube_a_pos[1], cube_a_pos[2], 1, 0, 0, 0
    ], dtype=float)

    data.qpos[cube_b_qpos_adr:cube_b_qpos_adr+7] = np.array([
        cube_b_pos[0], cube_b_pos[1], cube_b_pos[2], 1, 0, 0, 0
    ], dtype=float)

    # zero cube velocities too
    cube_a_qvel_adr = model.jnt_dofadr[get_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_A_JOINT)]
    cube_b_qvel_adr = model.jnt_dofadr[get_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_B_JOINT)]
    data.qvel[cube_a_qvel_adr:cube_a_qvel_adr+6] = 0.0
    data.qvel[cube_b_qvel_adr:cube_b_qvel_adr+6] = 0.0

    mujoco.mj_forward(model, data)


def sample_cube_positions(
    x_range=(0.40, 0.65),
    y_range=(-0.18, 0.18),
    z=0.245,
    min_separation=0.10,
):
    for _ in range(100):
        a = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            z
        ], dtype=float)

        b = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            z
        ], dtype=float)

        if np.linalg.norm(a[:2] - b[:2]) > min_separation:
            return a, b

    raise RuntimeError("Could not sample valid cube positions")


def run_fsm_episode(model, data, cube_a_init, cube_b_init, max_steps=2000, log_every=0):
    site_id = get_id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    cube_a_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_A_BODY)
    cube_b_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_B_BODY)

    cube_a_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_A_JOINT)
    cube_b_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_B_JOINT)
    cube_a_qpos_adr = model.jnt_qposadr[cube_a_joint_id]
    cube_b_qpos_adr = model.jnt_qposadr[cube_b_joint_id]

    reset_robot(model, data)
    reset_cubes(model, data, cube_a_qpos_adr, cube_b_qpos_adr, cube_a_init, cube_b_init)
    print("after reset:")
    print("cube_a actual:", data.xpos[cube_a_id].copy())
    print("cube_b actual:", data.xpos[cube_b_id].copy())

    home_q = HOME_CTRL[:ARM_DOF].copy()
    target_R = site_rotmat(data, site_id)
    safe_away_pos = np.array([0.45, 0.0, 0.55])

    state = "move_above_a"
    state_counter = 0
    pick_pos = cube_a_init.copy()
    place_pos = None

    hover_z = 0.14
    grasp_z = 0.00
    lift_z = 0.18
    place_z = 0.105
    retreat_z = 0.18

    traj_obs = []
    traj_actions = []
    traj_next_obs = []
    traj_dones = []

    done = False

    for step in range(max_steps):
        cube_a_pos = data.xpos[cube_a_id].copy()
        cube_b_pos = data.xpos[cube_b_id].copy()
        site_pos = data.site_xpos[site_id].copy()

        target_pos = site_pos.copy()
        grip_cmd = GRIP_OPEN

        if state == "move_above_a":
            target_pos = pick_pos + np.array([0.0, 0.0, hover_z])
            grip_cmd = GRIP_OPEN
            if near(site_pos, target_pos, tol=0.015):
                state = "lower_to_a"
                state_counter = 0

        elif state == "lower_to_a":
            target_pos = pick_pos + np.array([0.0, 0.0, grasp_z])
            grip_cmd = GRIP_OPEN
            if near(site_pos, target_pos, tol=0.008):
                state = "close_on_a"
                state_counter = 0

        elif state == "close_on_a":
            target_pos = pick_pos + np.array([0.0, 0.0, grasp_z])
            grip_cmd = GRIP_CLOSED
            state_counter += 1
            if state_counter > 120:
                state = "lift_a"
                state_counter = 0

        elif state == "lift_a":
            target_pos = pick_pos + np.array([0.0, 0.0, lift_z])
            grip_cmd = GRIP_CLOSED
            if near(site_pos, target_pos, tol=0.03):
                place_pos = cube_b_pos.copy()
                state = "move_above_b"
                state_counter = 0

        elif state == "move_above_b":
            target_pos = place_pos + np.array([0.0, 0.0, lift_z])
            grip_cmd = GRIP_CLOSED
            if near(site_pos, target_pos, tol=0.03):
                state = "lower_to_b"
                state_counter = 0

        elif state == "lower_to_b":
            target_pos = place_pos + np.array([0.0, 0.0, place_z])
            grip_cmd = GRIP_CLOSED
            if near(site_pos, target_pos, tol=0.03):
                state = "open_on_b"
                state_counter = 0

        elif state == "open_on_b":
            target_pos = place_pos + np.array([0.0, 0.0, place_z])
            grip_cmd = GRIP_OPEN
            state_counter += 1
            if state_counter > 100:
                state = "retreat"

        elif state == "retreat":
            target_pos = place_pos + np.array([0.0, 0.0, retreat_z])
            grip_cmd = GRIP_OPEN
            if near(site_pos, target_pos, tol=0.03):
                state = "return_home"

        elif state == "return_home":
            target_pos = safe_away_pos
            grip_cmd = GRIP_OPEN
            if near(site_pos, target_pos, tol=0.02):
                print("Task complete.")
                break

        obs = get_obs(model, data, site_id, cube_a_id, cube_b_id)

        q_des, pos_err, rot_err = pose_ik_step(
            model=model,
            data=data,
            site_id=site_id,
            target_pos=target_pos,
            target_R=target_R,
            home_q=home_q,
            step_size=0.35,
            damping=1e-2,
            rot_weight=2.0,
            posture_gain=0.03,
        )

        ctrl = data.ctrl.copy()
        ctrl[:ARM_DOF] = q_des
        ctrl[7] = grip_cmd
        action = ctrl.copy()

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        next_obs = get_obs(model, data, site_id, cube_a_id, cube_b_id)

        traj_obs.append(obs)
        traj_actions.append(action)
        traj_next_obs.append(next_obs)
        traj_dones.append(done)

        if log_every and step % log_every == 0:
            print(
                f"[{step:04d}] state={state:<12} "
                f"|pos_err|={np.linalg.norm(pos_err):.4f} "
                f"|rot_err|={np.linalg.norm(rot_err):.4f} "
                f"cube_a=({cube_a_pos[0]:.3f},{cube_a_pos[1]:.3f},{cube_a_pos[2]:.3f})"
            )

        if done:
            break

    success = check_success(data, cube_a_id, cube_b_id)

    return {
        "obs": np.array(traj_obs, dtype=np.float32),
        "actions": np.array(traj_actions, dtype=np.float32),
        "next_obs": np.array(traj_next_obs, dtype=np.float32),
        "dones": np.array(traj_dones, dtype=bool),
        "success": success,
        "cube_a_init": cube_a_init.astype(np.float32),
        "cube_b_init": cube_b_init.astype(np.float32),
        "episode_len": len(traj_obs),
    }


def collect_dataset(num_episodes=200, save_dir=SAVE_DIR, log_every=0):
    os.makedirs(save_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    saved = 0
    attempted = 0

    while saved < num_episodes:
        attempted += 1
        cube_a_init, cube_b_init = sample_cube_positions()

        traj = run_fsm_episode(
            model=model,
            data=data,
            cube_a_init=cube_a_init,
            cube_b_init=cube_b_init,
            max_steps=2000,
            log_every=log_every,
        )

        if traj["success"]:
            path = os.path.join(save_dir, f"traj_{saved:05d}.npz")
            np.savez(
                path,
                obs=traj["obs"],
                actions=traj["actions"],
                next_obs=traj["next_obs"],
                dones=traj["dones"],
                success=np.array([traj["success"]], dtype=bool),
                cube_a_init=traj["cube_a_init"],
                cube_b_init=traj["cube_b_init"],
                episode_len=np.array([traj["episode_len"]], dtype=np.int32),
            )
            saved += 1
            print(f"saved {path}  ({saved}/{num_episodes})")
        else:
            print(f"episode failed, skipping  (attempt {attempted})")

    print(f"done. saved {saved} successful trajectories.")


if __name__ == "__main__":
    collect_dataset(num_episodes=300, save_dir="datasets", log_every=0)