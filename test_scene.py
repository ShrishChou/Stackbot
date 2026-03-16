import time
import numpy as np
import mujoco
import mujoco.viewer

# model = mujoco.MjModel.from_xml_path("franka_emika_panda/stack_scene.xml")
model = mujoco.MjModel.from_xml_path("franka_emika_panda/mjx_single_cube.xml")
data = mujoco.MjData(model)
HOME_CTRL = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 255], dtype=float)
data.ctrl[:] = HOME_CTRL
mujoco.mj_forward(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()
    while viewer.is_running():
        t = time.time() - t0
        ctrl = HOME_CTRL.copy()
        if t > 2.0:
            ctrl[7] = 0
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)