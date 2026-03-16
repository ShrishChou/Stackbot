import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "franka_emika_panda/stack_scene.xml"

HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255], dtype=float)

ARM_DOF = 7
GRIP_OPEN = 255.0
GRIP_CLOSED = 0.0

SITE_NAME = "grasp_site"
CUBE_A_BODY = "cube_a"
CUBE_B_BODY = "cube_b"


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
    """
    6D pose IK:
      - 3D position tracking
      - 3D orientation tracking
      - nullspace posture term toward home_q
    """
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

    # nullspace posture term
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


def log_status(step, state, target_pos, current_pos, pos_err, rot_err, data):
    print(
        f"[{step:05d}] state={state:<12} "
        f"target=({target_pos[0]: .3f},{target_pos[1]: .3f},{target_pos[2]: .3f}) "
        f"curr=({current_pos[0]: .3f},{current_pos[1]: .3f},{current_pos[2]: .3f}) "
        f"|pos_err|={np.linalg.norm(pos_err): .4f} "
        f"|rot_err|={np.linalg.norm(rot_err): .4f} "
        f"finger_qpos=({data.qpos[7]:.4f},{data.qpos[8]:.4f}) "
        f"grip_ctrl={data.ctrl[7]:.1f}"
    )

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

site_id = get_id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
cube_a_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_A_BODY)
cube_b_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_B_BODY)

data.ctrl[:] = HOME_CTRL
mujoco.mj_forward(model, data)

for _ in range(300):
    data.ctrl[:] = HOME_CTRL
    mujoco.mj_step(model, data)

home_q = HOME_CTRL[:ARM_DOF].copy()

TARGET_R = site_rotmat(data, site_id)

state = "move_above_a"
last_state = None
state_counter = 0
step = 0

pick_pos = None
place_pos = None

hover_z = 0.14
grasp_z = 0.00
lift_z = 0.18
place_z = 0.105
retreat_z = 0.18

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step += 1

        cube_a_pos = data.xpos[cube_a_id].copy()
        cube_b_pos = data.xpos[cube_b_id].copy()
        site_pos = data.site_xpos[site_id].copy()

        if pick_pos is None:
            pick_pos = cube_a_pos.copy()

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
                state = "done"
                print("Task complete.")
                break

        

        q_des, pos_err, rot_err = pose_ik_step(
            model=model,
            data=data,
            site_id=site_id,
            target_pos=target_pos,
            target_R=TARGET_R,
            home_q=home_q,
            step_size=0.35,
            damping=1e-2,
            rot_weight=2.0,
            posture_gain=0.03,
        )

        ctrl = data.ctrl.copy()
        ctrl[:ARM_DOF] = q_des
        ctrl[7] = grip_cmd
        data.ctrl[:] = ctrl

        if state != last_state:
            print(f"\n=== STATE CHANGE: {last_state} -> {state} ===")
            last_state = state

        if step % 25 == 0:
            log_status(step, state, target_pos, site_pos, pos_err, rot_err, data)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)