import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "franka_emika_panda/stack_scene.xml"
DEFAULT_TRAJ = "datasets/traj_00209.npz"

HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255], dtype=float)

CUBE_A_JOINT = "cube_a_freejoint"
CUBE_B_JOINT = "cube_b_freejoint"
CUBE_A_BODY = "cube_a"
CUBE_B_BODY = "cube_b"


def get_id(model, objtype, name):
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx == -1:
        raise ValueError(f"Could not find {name}")
    return idx


def reset_robot(model, data):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if model.na > 0:
        data.act[:] = 0.0

    # 7 Panda joints + 2 finger joints
    data.qpos[:9] = np.array(
        [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04],
        dtype=float,
    )
    data.ctrl[:] = HOME_CTRL
    mujoco.mj_forward(model, data)

    # settle
    for _ in range(200):
        data.ctrl[:] = HOME_CTRL
        mujoco.mj_step(model, data)


def reset_cubes(model, data, cube_a_pos, cube_b_pos):
    cube_a_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_A_JOINT)
    cube_b_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_B_JOINT)

    cube_a_qpos_adr = model.jnt_qposadr[cube_a_joint_id]
    cube_b_qpos_adr = model.jnt_qposadr[cube_b_joint_id]

    cube_a_qvel_adr = model.jnt_dofadr[cube_a_joint_id]
    cube_b_qvel_adr = model.jnt_dofadr[cube_b_joint_id]

    # freejoint qpos = x y z qw qx qy qz
    data.qpos[cube_a_qpos_adr:cube_a_qpos_adr + 7] = np.array(
        [cube_a_pos[0], cube_a_pos[1], cube_a_pos[2], 1, 0, 0, 0],
        dtype=float,
    )
    data.qpos[cube_b_qpos_adr:cube_b_qpos_adr + 7] = np.array(
        [cube_b_pos[0], cube_b_pos[1], cube_b_pos[2], 1, 0, 0, 0],
        dtype=float,
    )

    data.qvel[cube_a_qvel_adr:cube_a_qvel_adr + 6] = 0.0
    data.qvel[cube_b_qvel_adr:cube_b_qvel_adr + 6] = 0.0

    mujoco.mj_forward(model, data)


def check_stack_success(data, model):
    cube_a_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_A_BODY)
    cube_b_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, CUBE_B_BODY)

    cube_a = data.xpos[cube_a_id].copy()
    cube_b = data.xpos[cube_b_id].copy()

    xy_dist = np.linalg.norm(cube_a[:2] - cube_b[:2])
    dz = cube_a[2] - cube_b[2]

    xy_ok = xy_dist < 0.03
    z_ok = abs(dz - 0.05) < 0.03

    return xy_ok and z_ok, xy_dist, dz


def main():
    traj_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRAJ
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    traj = np.load(traj_path, allow_pickle=False)

    if "actions" not in traj:
        raise KeyError("Trajectory missing 'actions'")
    if "cube_a_init" not in traj or "cube_b_init" not in traj:
        raise KeyError("Trajectory missing 'cube_a_init' or 'cube_b_init'")

    actions = traj["actions"]
    cube_a_init = traj["cube_a_init"]
    cube_b_init = traj["cube_b_init"]

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    reset_robot(model, data)
    reset_cubes(model, data, cube_a_init, cube_b_init)

    print(f"Loaded {traj_path}")
    print(f"Trajectory length: {len(actions)}")
    print(f"cube_a_init: {cube_a_init}")
    print(f"cube_b_init: {cube_b_init}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # small pause so you can inspect initial state
        for _ in range(50):
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        for t, action in enumerate(actions):
            data.ctrl[:] = action
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

        success, xy_dist, dz = check_stack_success(data, model)
        print(f"Replay finished. success={success}, xy_dist={xy_dist:.4f}, dz={dz:.4f}")

        # hold final frame for inspection
        while viewer.is_running():
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()