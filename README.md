# Stackbot

A self-contained imitation learning pipeline for robotic object stacking, simulated in MuJoCo using a Franka Emika Panda arm. A deterministic finite state machine (FSM) acts as the expert controller, collecting demonstration trajectories with randomized object perturbations. Those trajectories train a feedforward neural network via behavior cloning to reproduce the stacking behavior by predicting end-effector deltas at each timestep.

---

## Pipeline

```
FSM.py                    →   FSM_collection.py            →   Imitation_learn.py         →   evals.py
─────────────────────────     ──────────────────────────       ────────────────────────       ───────────────────────
Deterministic expert          Batch N episodes with             Train MLP on (obs → Δ)        Generate heatmaps to
controller with 6D pose IK    random object perturbations       via behavior cloning          evaluate policy in MuJoCo
```

---

## Repository Structure

```
Stackbot/
├── franka_emika_panda/       # Franka Panda MJCF model + stack_scene.xml
├── FSM.py                    # FSM expert: states, IK solver, viewer
├── FSM_collection.py         # Batched data collection with random perturbations
├── Imitation_learn.py        # MLP architecture + behavior cloning training
├── run_policy.py             # Load trained model, roll out in MuJoCo
├── test_scene.py             # Scene setup / sanity checks
├── validate_dataset.py       # Dataset inspection and validation
└── vis_data.py               # Trajectory visualization
```

---

## FSM Expert (`FSM.py`)

The expert is built around a Franka Panda arm controlled via **Jacobian-based 6D pose IK** — solving for joint deltas that minimize both position and orientation error at the `grasp_site`, with a nullspace posture term keeping the arm near its home configuration.

### States

| State | Description |
|---|---|
| `move_above_a` | Move end-effector above cube A at hover height |
| `lower_to_a` | Descend to grasp height above cube A |
| `close_on_a` | Close gripper to grasp cube A (timed hold) |
| `lift_a` | Lift cube A to safe travel height |
| `move_above_b` | Translate above cube B while holding cube A |
| `lower_to_b` | Descend until cube A is seated on cube B |
| `open_on_b` | Release gripper over cube B (timed hold) |
| `retreat` | Raise end-effector away from the stack |
| `return_home` | Return to safe home position |

### IK Solver

At each timestep, the solver computes:

```
J = [Jp ; rot_weight * Jr]          # stacked position + orientation Jacobian
dq_task = Jᵀ (JJᵀ + λI)⁻¹ err      # damped least-squares task velocity
dq = dq_task + (I - J⁺J) * dq_posture  # + nullspace posture term
q_des = q + step_size * dq
```

Key parameters: `step_size=0.35`, `damping=1e-2`, `rot_weight=2.0`, `posture_gain=0.03`.

---

## Data Collection (`FSM_collection.py`)

Runs the FSM across many episodes with **random XY perturbations** applied to the initial positions of cube A and cube B. Each episode logs a trajectory of `(observation, delta_action)` pairs:

- **Observation**: end-effector position, grasp site orientation, gripper state, cube A pose, cube B pose
- **Delta action**: `Δ = [Δx, Δy, Δz, Δgripper]` — the change in end-effector pose from step `t` to `t+1`

Datasets are saved to a `datasets/` folder and run headlessly using `mjpython` to avoid the MuJoCo viewer overhead during batch collection.

---

## Imitation Learning (`Imitation_learn.py`)

Trains a feedforward MLP via **behavior cloning** on the collected demonstration dataset.

### Architecture

Three repeated blocks of:
```
Linear(in → 256) → ReLU → Dropout
```
followed by a final `Linear(256 → action_dim)` output head.

- **Input**: observation vector
- **Output**: `[Δx, Δy, Δz, Δgripper]`
- **Loss**: MSE between predicted and expert deltas
- **Training**: supervised, loads all `.pkl` files from `datasets/`

---

## Running the Learned Policy (`run_policy.py`)

Loads a trained model checkpoint and runs it inside MuJoCo. At each timestep the policy receives the current observation, predicts a delta, applies it to the end-effector, and steps the simulation — reconstructing the stacking trajectory from learned motion primitives.

```bash
mjpython run_policy.py
```

---

## Requirements

```bash
pip install mujoco numpy torch
```

- Python 3.10+
- MuJoCo 3.0+
- PyTorch
- `mjpython` (ships with MuJoCo) — required for headless simulation during data collection

---

## Usage

### 1. Clone

```bash
git clone https://github.com/ShrishChou/Stackbot.git
cd Stackbot
```

### 2. Verify the scene

```bash
mjpython test_scene.py
```

### 3. Watch the FSM expert

```bash
mjpython FSM.py
```

### 4. Collect demonstrations

```bash
mjpython FSM_collection.py
```

Trajectories are saved to `datasets/`.

### 5. Validate dataset (optional)

```bash
python validate_dataset.py
python vis_data.py
```

### 6. Train

```bash
python Imitation_learn.py
```

### 7. Run the learned policy

```bash
mjpython run_policy.py
```

---

## Key Design Choices

**FSM as expert.** Programmatic demonstration generation produces clean, repeatable, noise-free trajectories without human teleoperation. It also decouples data quality from the learning problem — if the policy fails to generalize, the issue is with the model, not the data.

**Random perturbations.** Varying cube positions across episodes prevents the model from memorizing absolute coordinates and forces it to learn the relative geometry of the task.

**Delta actions.** Representing actions as `Δ` rather than absolute targets makes the learned policy translatable — the network learns *how to move* toward a goal rather than hard-coding workspace coordinates.

**`mjpython` for headless collection.** Using MuJoCo's bundled `mjpython` interpreter enables fast, viewer-free batch simulation during data collection without needing a separate rendering context.
