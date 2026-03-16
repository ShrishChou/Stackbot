import numpy as np
import torch
import torch.nn as nn
import mujoco
import time
ckpt = torch.load("bc_runs/bc_policy_best.pt", map_location="cpu")
predict_delta = ckpt["predict_delta"]
MODEL_PATH = "franka_emika_panda/stack_scene.xml"

POLICY_PATH = "bc_runs/bc_policy_best.pt"
STATS_PATH = "bc_runs/normalization_stats.npz"

SITE_NAME = "grasp_site"
CUBE_A_BODY = "cube_a"
CUBE_B_BODY = "cube_b"

ARM_DOF = 7


# =========================
# Model definition
# =========================

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Helper functions
# =========================

def get_id(model, objtype, name):
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx == -1:
        raise ValueError(f"Could not find {name}")
    return idx


def get_obs(model, data, site_id, cube_a_id, cube_b_id):

    site_pos = data.site_xpos[site_id].copy()
    cube_a_pos = data.xpos[cube_a_id].copy()
    cube_b_pos = data.xpos[cube_b_id].copy()

    obs = np.concatenate([
        data.qpos[:7],
        data.qvel[:7],
        data.qpos[7:9],
        site_pos,
        cube_a_pos,
        cube_b_pos,
        cube_a_pos - site_pos,
        cube_b_pos - cube_a_pos,
    ])

    return obs.astype(np.float32)


def sample_cube_positions():

    x_range = (0.40, 0.65)
    y_range = (-0.18, 0.18)
    z = 0.245

    while True:

        a = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            z
        ])

        b = np.array([
            np.random.uniform(*x_range),
            np.random.uniform(*y_range),
            z
        ])

        if np.linalg.norm(a[:2] - b[:2]) > 0.10:
            return a, b


def reset_robot(model, data):

    data.qpos[:] = 0
    data.qvel[:] = 0

    data.qpos[:9] = np.array(
        [0,0,0,-1.57079,0,1.57079,-0.7853,0.04,0.04]
    )

    mujoco.mj_forward(model,data)


def reset_cubes(model,data,cube_a_pos,cube_b_pos):

    cube_a_joint = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"cube_a_freejoint")
    cube_b_joint = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"cube_b_freejoint")

    adr_a = model.jnt_qposadr[cube_a_joint]
    adr_b = model.jnt_qposadr[cube_b_joint]

    data.qpos[adr_a:adr_a+7] = np.array([
        cube_a_pos[0],
        cube_a_pos[1],
        cube_a_pos[2],
        1,0,0,0
    ])

    data.qpos[adr_b:adr_b+7] = np.array([
        cube_b_pos[0],
        cube_b_pos[1],
        cube_b_pos[2],
        1,0,0,0
    ])

    mujoco.mj_forward(model,data)


# =========================
# Load trained policy
# =========================

ckpt = torch.load(POLICY_PATH,map_location="cpu")
stats = np.load(STATS_PATH)

obs_mean = stats["obs_mean"]
obs_std = stats["obs_std"]

act_mean = stats["act_mean"]
act_std = stats["act_std"]

obs_dim = ckpt["obs_dim"]
act_dim = ckpt["act_dim"]

model = MLPPolicy(obs_dim,act_dim)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# =========================
# Mujoco setup
# =========================

model_mj = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model_mj)

site_id = get_id(model_mj,mujoco.mjtObj.mjOBJ_SITE,SITE_NAME)
cube_a_id = get_id(model_mj,mujoco.mjtObj.mjOBJ_BODY,CUBE_A_BODY)
cube_b_id = get_id(model_mj,mujoco.mjtObj.mjOBJ_BODY,CUBE_B_BODY)

viewer = mujoco.viewer.launch_passive(model_mj,data)

def check_success(data, cube_a_id, cube_b_id, xy_tol=0.03, z_tol=0.03, expected_dz=0.05):
    cube_a = data.xpos[cube_a_id].copy()
    cube_b = data.xpos[cube_b_id].copy()

    xy_ok = np.linalg.norm(cube_a[:2] - cube_b[:2]) < xy_tol
    z_ok = abs((cube_a[2] - cube_b[2]) - expected_dz) < z_tol
    return xy_ok and z_ok

# =========================
# Run policy
# =========================

cube_a,cube_b = sample_cube_positions()

reset_robot(model_mj,data)
reset_cubes(model_mj,data,cube_a,cube_b)

print("Cube A:",cube_a)
print("Cube B:",cube_b)

success_counter = 0
SUCCESS_HOLD_STEPS = 30
done = False
hold_q = None
while viewer.is_running():

    if not done:
        obs = get_obs(model_mj, data, site_id, cube_a_id, cube_b_id)
        obs_norm = (obs - obs_mean) / obs_std
        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_norm = model(obs_tensor).cpu().numpy()[0]

        action = action_norm * act_std + act_mean

        if predict_delta:
            action[:7] = obs[:7] + action[:7]

        # smooth arm only
        data.ctrl[:7] = 0.8 * data.ctrl[:7] + 0.2 * action[:7]

        # threshold gripper
        data.ctrl[7] = 255.0 if action[7] > 127.5 else 0.0

        # clamp arm
        for i in range(7):
            low, high = model_mj.actuator_ctrlrange[i]
            data.ctrl[i] = np.clip(data.ctrl[i], low, high)

        if check_success(data, cube_a_id, cube_b_id):
            success_counter += 1
        else:
            success_counter = 0

        if success_counter >= SUCCESS_HOLD_STEPS:
            done = True
            hold_q = data.qpos[:7].copy()
            print("Success achieved. Holding final pose.")
            time.sleep(2.0)
            break

    else:
        data.ctrl[:7] = hold_q
        data.ctrl[7] = 255.0

    mujoco.mj_step(model_mj, data)
    viewer.sync()
    time.sleep(0.002)